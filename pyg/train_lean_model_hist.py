#!/usr/bin/env python

import argparse
import os, datetime, configargparse, logging
import wandb
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric
import torchvision

from model import MyFCModule, MyGraphSAGE_WithImage
from dataset import OSMChengduHist

from tqdm import tqdm

def gpu_stats():
    t = torch.cuda.get_device_properties(0).total_memory/1024**3
    r = torch.cuda.memory_reserved(0)/1024**3
    a = torch.cuda.memory_allocated(0)/1024**3
    f = r-a  # free inside reserved
    logging.info(f'total: {t:.2f} GB, reserved {r:.2f} GB, allocated {a:.2f} GB, free {f:.2f} GB')

def calc_acc(pred, target):
    correct = (pred == target).sum()
    return int(correct) / len(target)
    

def save_checkpoint(model, checkpoint_dir, save_best=False):
    last_path = os.path.join(checkpoint_dir,f'model_last.pth')
    torch.save(model, last_path)
    logging.info("Saving last model : model_best.pth")
    if save_best:
        best_path = os.path.join(checkpoint_dir,'model_best.pth')
        torch.save(model, best_path)
        logging.info("Saving current best: model_best.pth")

def log_dir(args):
    if args.wandb_log:
        log_dir = os.path.join(args.wandb_dir, 'runs', wandb.run.project, f'{args.timestamp:%Y%m%d_%H%M%S}_{wandb.run.name}')
    else:
        log_dir = os.path.join(args.wandb_dir, 'runs', args.wandb_project, f'{args.timestamp:%Y%m%d_%H%M%S}_unnamed')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return log_dir


def main(args, loglevel):
    args.timestamp = datetime.datetime.now()

    # Setting random seeds
    if args.random_seed == -1:
        args.random_seed = np.random.randint(0,100)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Initialize W&B experiment
    if args.wandb_log:
        tags = ['debug']

        if args.tag is not None:
            tags.append(args.tag)
        
        
        tags.append('no_backbone')
        

        run = wandb.init(project=args.wandb_project, entity="ostromann", tags=tags, dir=args.wandb_dir)
        wandb.config.update(args)

    args.save_dir = log_dir(args)

    if args.wandb_log:
        wandb.config.update(args, allow_val_change=True)

    # Configure Logger
    logging.basicConfig(format='%(levelname)-6s\t %(asctime)s %(message)s',
                        level=loglevel,
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                        logging.FileHandler(os.path.join(args.save_dir, f'{args.timestamp:%Y-%m-%d_%H:%M}.log')),
                        logging.StreamHandler()]
                        )

    logging.info(f'Reading graph from {args.path}')
    data = OSMChengduHist(args.path, hist_bins=args.hist_bins)[0]
    logging.info(f'Read data from graph: {data}')

    if args.debug:
        debug_loader = torch_geometric.loader.NeighborLoader(
        data,
        num_neighbors=[args.n_neighbors_1, args.n_neighbors_2],
        batch_size=10000, 
        shuffle=False)
        first_batch = None
        for batch in debug_loader:
            if not first_batch:
                first_batch = batch
            pass
        data = first_batch
        logging.info(f'DEBUGGING MODE: Reduced dataset: {data}')
       
    n_node_attrs = data.x.shape[1]
    n_classes = torch.unique(data.y).shape[0]+1

    if not torch.cuda.is_available:
        logging.warn(f'CUDA not available to torch!!! Running on CPU instead!')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    visual_features = args.hist_bins*3
    model = MyGraphSAGE_WithImage(n_node_attrs + visual_features, args.gcn_hidden_units, args.gcn_embed_dim, n_classes, args.gcn_dropout_rate)
   
    if args.fc_only:
        logging.warning(f'fc_only should only be used for debugging!')
        model = MyFCModule(n_node_attrs, args.gcn_hidden_units, n_classes)

    if args.full_batch:
        args.batch_size = int(data.train_mask.sum())
        if args.wandb_log:
            wandb.config.update(args, allow_val_change=True)

    train_loader = torch_geometric.loader.NeighborLoader(
        data,
        num_neighbors=[args.n_neighbors_1, args.n_neighbors_2],
        batch_size=args.batch_size,
        input_nodes=data.train_mask, 
        shuffle=True)

    eval_loader = torch_geometric.loader.NeighborLoader(
        data,
        num_neighbors=[args.n_neighbors_1, args.n_neighbors_2],
        batch_size=args.batch_size, 
        shuffle=True)

    if args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError('optimizer must be \'sgd\' or \'adam\'!')

    if args.lr_scheduler is not None:
        if args.lr_scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)
        elif args.lr_scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma) # default gamma=0.1
        elif args.lr_scheduler == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_scheduler_gamma) # default gamma=0.9
        elif args.lr_scheduler == 'None':
            args.lr_scheduler =  None
            if args.wandb_log:
                wandb.config.update(args, allow_val_change=True)
        else:
            raise ValueError(f'invalid argument for lr_scheduler. Is {args.lr_scheduler}, but must be in [ReduceLROnPlateau, ExponentialLR, StepLR]')

    best_val_acc = 0.0
    test_acc_at_best_epoch = 0.0
    best_epoch = 0
    current_lr = args.learning_rate

    # Create output file for test stats and write header
    test_stats_file = os.path.join(args.save_dir,"test_stats.csv")
    with open(test_stats_file, "w") as f:
        f.write('epoch; train_loss; val_loss; test_loss; train_acc; val_acc; test_acc; best_epoch; test_acc_at_best_epoch')

    # Training Loop
    for epoch in range(args.epochs):
        best = False

        iter= 0
        for batch in train_loader:
            # Send data and model to device
            batch = batch.to(device)
            model = model.to(device)
            model.train()
            optimizer.zero_grad()
            out = model(batch)

            # gpu_stats()
            loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()

            if iter % args.verbose_every == 0:
                logging.info(f'[{epoch+1:04d}/{args.epochs:04d}] {iter} of {len(train_loader)}: train_loss: {loss.item()}')
                if args.wandb_log:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': loss,
                        })
            iter += 1
        
        # TODO: Does this really help to clear the GPU?
        model = model.to('cpu')
        batch = batch.to('cpu')
        torch.cuda.empty_cache()

        # Evaluate
        iter = 0
        if (epoch+1) % args.eval_every == 0:
            logging.info('Evaluating...')
            # Set model to eval
            model.eval()

            # 1) Too slow to do the evaluation on CPU
            # 2) On GPU storing appending the output to outs leads to VRAM overflow
            # 3) Now, calculate correct samples and length of each batch to calculated accuracy later
            val_correct = 0
            val_size = 0
            train_correct = 0
            train_size = 0
            test_correct = 0
            test_size = 0

            train_loss = 0
            val_loss = 0
            test_loss = 0

            for batch in tqdm(eval_loader, total=len(eval_loader)):
                batch = batch.to(device)
                model = model.to(device)
                out = model(batch)[:batch.batch_size]
                pred = out.argmax(dim=1)

                # Only use original input nodes
                batch.y = batch.y[:batch.batch_size]
                batch.val_mask = batch.val_mask[:batch.batch_size]
                batch.train_mask = batch.train_mask[:batch.batch_size]
                batch.test_mask = batch.test_mask[:batch.batch_size]

                # Calculate avg loss per sample
                train_loss += F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask]).item()/batch.train_mask.sum() if batch.train_mask.sum() > 0 else 0
                val_loss += F.nll_loss(out[batch.val_mask], batch.y[batch.val_mask]).item()/batch.val_mask.sum() if batch.val_mask.sum() > 0 else 0
                test_loss += F.nll_loss(out[batch.test_mask], batch.y[batch.test_mask]).item()/batch.test_mask.sum() if batch.test_mask.sum() > 0 else 0

                # Count correct and total number of samples per train, val and test sets
                val_correct += (pred[batch.val_mask] == batch.y[batch.val_mask]).sum()
                val_size += len(batch.y[batch.val_mask])
                train_correct += (pred[batch.train_mask] == batch.y[batch.train_mask]).sum()
                train_size += len(batch.y[batch.train_mask])
                test_correct += (pred[batch.test_mask] == batch.y[batch.test_mask]).sum()
                test_size += len(batch.y[batch.test_mask])

                if iter % args.verbose_every == 0:
                    logging.info(f' val: {val_correct}/{val_size} - train: {train_correct}/{train_size} - test: xxx/{test_size}')
                iter +=1
                
            logging.info(f'Final evaluation this epoch: val: {val_correct}/{val_size} - train: {train_correct}/{train_size} - test {data.test_mask.sum()}')
            val_acc = val_correct / val_size
            train_acc = train_correct / train_size
            test_acc = test_correct / test_size
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best = True
                test_acc_at_best_epoch = test_acc
                best_epoch = epoch
                logging.info('***new best model***')

            # Save Checkpoints
            save_checkpoint(model, args.save_dir, save_best=best)

            # Logging
            if (epoch+1) % args.log_every == 0:
                logging.info(f'[{epoch+1:04d}/{args.epochs:04d}]  '\
                    f'loss: {train_loss:8.4f}  '\
                    f'train_acc: {train_acc:.3f}  '\
                    f'val_loss: {val_loss:.3f}  '\
                    f'val_acc: {val_acc:.3f} '\
                    f'best_val_acc: {best_val_acc:.3f} '\
                    f'test_acc_@_best_epoch: {test_acc_at_best_epoch:.3f} '\
                    f'best_epoch: {best_epoch}')

                if args.wandb_log:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss, 
                        'train_acc': train_acc, 
                        'val_loss': val_loss, 
                        'val_acc': val_acc,
                        'best_val_acc': best_val_acc,
                        'best_epoch': best_epoch,
                        'lr': current_lr,
                        })

            # Writing full stats
            with open(test_stats_file, "a") as f:
                f.write(f'\n{epoch}; {train_loss:.4f}; {val_loss:.4f}; {test_loss:.4f}; {train_acc:.3f}; {val_acc:.3f}; {test_acc:.3f}; {best_epoch}; {test_acc_at_best_epoch:.3f}')

        # Adjust learning rate
        if args.lr_scheduler:
            scheduler.step()
            last_lr = current_lr
            current_lr = scheduler.get_last_lr()[0]
            if last_lr != current_lr:
                logging.info(f'learning rate adjusted, now {current_lr} (was {last_lr})')

    
    logging.info(f'Training finished...')
    logging.info(f'Test stats written to {test_stats_file} . Don\'t peak!') 

    if args.wandb_log:
        logging.info('Uploading test stats to W&B...')
        artifact = wandb.Artifact(run.name + '-test_stats', type='experiment-stats')
        # Add a file to the artifact's contents
        artifact.add_file(test_stats_file)
        # Save the artifact version to W&B and mark it as the output of this run
        run.log_artifact(artifact)

        logging.info('Uploading best model to W&B...')
        artifact = wandb.Artifact(run.name + '-model_best', type='model')
        # Add a file to the artifact's contents
        artifact.add_file(os.path.join(args.save_dir,'model_best.pth'))
        # Save the artifact version to W&B and mark it as the output of this run
        run.log_artifact(artifact)
        



if __name__ == '__main__':
    p = configargparse.ArgParser( default_config_files=[''],
                                    description = "Training script for CNN-GCN models")

    # p.add("path", default ='./data/osm_20211104', help = "path to training data", metavar = "ARG")
    p.add('--path', default ='osm_20211117/', help = "path to training data")
    p.add('-c', '--config', required=False, is_config_file=True, help='config file path')
    p.add('-v', '--verbose', action='store_true', help="increase output verbosity")
    p.add('--debug', default=False, action='store_true', help='debug mode')

    # Setup args
    p.add('--random_seed', default=42, type=int, help='Seed to random number generators (set to -1 (default) for random integer between 0 and 100)')
    
    p.add('--fc_only', default=False, action='store_true', help='use only an FC network (NO GCN, NO CNN!)')

    # VFE args
    p.add('--hist_bins', type=int, default=32, help='number of bins for colour intensity histograms')

    # GCN args
    p.add('--gcn', default=None, help='GCN layers to use (choose from [gcn, graphsage, gat])')
    p.add('--gcn_layers', default=None, type=int, help='number of GCN layers')
    p.add('--gcn_hidden_units', default='256', type=int, help='number of hidden units in GCN layers')
    p.add('--gcn_embed_dim', default=64, type=int, help='embedding dimensionality of GCN layers')
    p.add('--gcn_dropout_rate', default=0.0, type=float, help='probability for dropout in GCN layers')

    # Training args
    p.add('--epochs', default=20, type=int, help='total number of epochs')
    p.add('--optimizer', '--opt', default='sgd', help='choose optimizer \'sgd\' or  \'adam\'')
    p.add('--learning_rate','--lr', default=0.01, type=float, help='learning rate')
    p.add('--weight_decay', default=0.0, type=float, help='weight decay')
    p.add('--lr_scheduler', default=None, type=str, help='Choose a learning rate scheduler from [ReduceLROnPlateau, ExponentialLR, StepLR]')
    p.add('--lr_scheduler_gamma', default=0.1, type=float, help='decay rate for learning rate scheduler')
    p.add('--lr_scheduler_step_size', default= 20, type=int, help='stepsize for the StepLR')

    p.add('--eval_every', default=1, type=int, help='how often to evaluate model on validation and training data')

    # Batch args
    p.add('--n_neighbors_1', default=5, type=int, help='number of first-hop neighbors to be sampled')
    p.add('--n_neighbors_2', default=10, type=int, help='number of second-hop neighbors to be sampled')
    p.add('--batch_size', default=64, type=int, help='number of samples per batch')
    p.add('--full_batch', action='store_true', help='train on entire data. NOTE: Overwrites \'batch_size\'.')
    p.add('--val_batch_size', default=200, type=int, help='number of samples per validation batch')

    # Logging args
    p.add('--wandb_log', default=False, action='store_true', help='Activate logging to W&B')
    p.add('--wandb_dir', default='../../../../../proj/cvl/users/x_olist/experiments')
    p.add('--wandb_project', default='gcpr22', help='Project name on W&B')
    p.add('--log_every', default=1, type=int, help='how often to log to stdout and W&B')
    p.add('--verbose_every', default= 20, type=int, help='how often to verbose something during training')
    p.add('--tag', default=None, type=str, help='additional tag for the wandb logging.')

    # Output args
    p.add('--dest', default='./', help='path to destination folder')
    
    args = p.parse_args()
  
    # Setup logging
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    main(args, loglevel)