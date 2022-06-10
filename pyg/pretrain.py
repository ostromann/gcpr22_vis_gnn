from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import datetime
import os
import configargparse
import logging
import copy
import wandb
# logging.info("PyTorch Version: ",torch.__version__)
# logging.info("Torchvision Version: ",torchvision.__version__)


def train_model(model, dataloaders, criterion, optimizer, epochs=25, is_inception=False, device='cpu'):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(args.epochs):
        logging.info(f'Epoch {epoch}/{args.epochs - 1}')
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            logging.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

            if args.wandb_log:
                if phase == 'val':
                    wandb.log({
                        'epoch': epoch,
                        'val_loss': epoch_loss, 'val_acc': epoch_acc
                        })
                else:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': epoch_loss, 'train_acc': epoch_acc
                        })


    time_elapsed = time.time() - since
    logging.info(f'Training complete in {time_elapsed// 60:.0f}m {time_elapsed % 60:.0f}s')
    logging.info(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(backbone, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model = None
    input_size = 0

    if backbone == "resnet18":
        """ Resnet18
        """
        model = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif backbone == "resnet50":
        """ Resnet50
        """
        model = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif backbone == "alexnet":
        """ Alexnet
        """
        model = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif backbone == "vgg":
        """ VGG11_bn
        """
        model = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif backbone == "squeezenet":
        """ Squeezenet
        """
        model = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
        input_size = 224

    elif backbone == "densenet":
        """ Densenet
        """
        model = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif backbone == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        # Handle the auxilary net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        logging.info("Invalid model name, exiting...")
        exit()

    return model, input_size


def log_dir(args):
    if args.wandb_log:
        log_dir = os.path.join(args.wandb_dir, 'pretrain_runs', wandb.run.project, f'{args.timestamp:%Y%m%d_%H%M%S}_{wandb.run.name}')
    else:
        log_dir = os.path.join(args.wandb_dir, 'pretrain_runs', 'cnn_gcn', f'{args.timestamp:%Y%m%d_%H%M%S}_unnamed')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    return log_dir

def main(args, loglevel):
    args.timestamp = datetime.datetime.now()

    if args.wandb_log:
        run = wandb.init(project="cnn_gcn", entity="ostromann", tags=["nwpu-resisc-45", "pretrain", "debug"], dir=args.wandb_dir,  job_type='pretrain')
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

    logging.info('test')
    logging.info(logging)
    

    # TODO: Remove hard-coded stuff
    num_classes = 45

    # Initialize the model for this run
    model, input_size = initialize_model(args.backbone, num_classes, feature_extract=not args.fine_tune, use_pretrained=True)

    # Print the model we just instantiated
    logging.info(model)


    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    logging.info("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(args.path, x), data_transforms[x]) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Send the model to GPU
    model = model.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model.parameters()
    logging.info("Params to learn:")
    if not args.fine_tune:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                logging.info(f"\t{name}")
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                logging.info(f"\t{name}")

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=args.learning_rate, momentum=args.momentum)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model, hist = train_model(model, dataloaders_dict, criterion, optimizer_ft, epochs=args.epochs, is_inception=(args.backbone=="inception"), device=device)
    
    
    # get existing pre-trained model
    existing_artifact = wandb.Api().artifact("ostromann/cnn_gcn/" + args.backbone + '_on_' + args.path.split('/')[-1] + ':latest')

    # overwrite if better
    if hist[-1] > existing_artifact.metadata["val_acc"]:
        model_save_path = os.path.join(args.save_dir, args.backbone + '_on_' + args.path.split('/')[-1])
        torch.save(model, model_save_path)
        logging.info("Saving current best: model_best.pth")


        if args.wandb_log:
            logging.info('Uploading best model to W&B...')
            artifact = wandb.Artifact(args.backbone + '_on_' + args.path.split('/')[-1], type='pretrain_model', metadata={'val_acc': hist[-1]})
            # Add a file to the artifact's contents
            artifact.add_file(model_save_path)
            # Save the artifact version to W&B and mark it as the output of this run
            run.log_artifact(artifact)


if __name__ == '__main__':
    p = configargparse.ArgParser( default_config_files=[''],
                                    description = "Training script for backbone networks")

    # p.add("path", default ='./data/osm_20211104', help = "path to training data", metavar = "ARG")
    # p.add('--path', default ='./data/hymenoptera_data', help = "path to training data")
    p.add('--path', default='../../../datasets/NWPU-RESISC45_split_v2')
    p.add('-c', '--config', required=False, is_config_file=True, help='config file path')
    p.add('-v', '--verbose', action='store_true', help="increase output verbosity")


    # backbone args
    p.add('--backbone', default='resnet18', help='backbone network to use (choose from [resnet18, resnet50])')
    # p.add('--feature_extract', default=False, action='store_true', help='use backbone as feature extractor only (learn weights only on classifier head')
    p.add('--fine_tune', default=True, action='store_true', help='if True: don\'t freeze backbone, but instead fine tune whole model on new data.')

    # Training args
    p.add('--epochs', default=2, type=int, help='total number of epochs')
    p.add('--learning_rate','--lr', default=0.001, type=float, help='learning rate')
    p.add('--momentum', default=0.9, type=float, help='momentum')

    # Batch args
    p.add('--batch_size', default=512, type=int, help='number of samples per batch')

    # Logging args
    p.add('--wandb_log', default=True, action='store_true', help='Activate logging to W&B')
    p.add('--wandb_dir', default='../../../../../proj/cvl/users/x_olist/pre_train')
    
    args = p.parse_args()
  
    # Setup logging
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    main(args, loglevel)