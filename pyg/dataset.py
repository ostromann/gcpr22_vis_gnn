import torch
import numpy as np
import os, zipfile, logging, ast
from collections import defaultdict
import torch_geometric
from torch_geometric.data import InMemoryDataset
import networkx as nx
from tqdm import tqdm
import torchvision

class OSMChengduRawImage(InMemoryDataset):
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(OSMChengduRawImage, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # List of the raw files
        return ['srn_processed.graphml', 'image_data.zip']

    @property
    def processed_file_names(self):
        return 'osm_chengdu.pt'

    def download(self):
        pass

    def process(self):
        self.valid_node_attrs = [ 'bridge', 'oneway','centroid', 'bearing', 'length', 'translated_geometry']
        self.img_node_attrs = ['ortho']
        self.stringified_node_attrs = ['centroid','translated_geometry']
        self.node_label = 'highway'


        graph_path = os.path.join(self.raw_dir, 'srn_processed.graphml')
        G = nx.read_graphml(graph_path)

        logging.debug(f'Read graph from {graph_path}: {nx.info(G)}')

        G = nx.convert_node_labels_to_integers(G)
        G = G.to_directed() if not nx.is_directed(G) else G
        edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

        if G.number_of_nodes() > 0:
            node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
        else:
            node_attrs = {}

        if G.number_of_edges() > 0:
            edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
        else:
            edge_attrs = {}

        logging.debug(f'{len(node_attrs)} node attributes: {node_attrs}')
        logging.debug(f'{len(edge_attrs)} edge attributes: {edge_attrs}')

        data = defaultdict(list)
        zip_path = os.path.join(self.raw_dir, 'image_data.zip')
        with zipfile.ZipFile(zip_path, 'r') as myzip:
            logging.info(f'Converting node attributes to data dict...')
            for i, (n, feat_dict) in tqdm(enumerate(G.nodes(data=True)), total=G.number_of_nodes()):
                if set(feat_dict.keys()) != set(node_attrs):
                    logging.warning(f'Mismatching attributes for node {n}: got {feat_dict.keys()} expected {node_attrs}')
                    raise ValueError('Not all nodes contain the same attributes')

                for key, value in feat_dict.items():
                    if key in self.valid_node_attrs or key in self.img_node_attrs:

                        # Stringified attributes (lists and tuples)
                        if key in self.stringified_node_attrs:
                            data[str(key)].append(ast.literal_eval(value))
                        
                        # Image attributes 
                        elif key in self.img_node_attrs:
                            # Read corresponding image patch from zip file
                            tmp_arr = np.load(myzip.extract(value, path='/tmp')) 

                            if tmp_arr.dtype == np.uint16:
                                tmp_arr = tmp_arr.astype(np.int16)

                            data[str(key)].append(tmp_arr)
                            os.remove(os.path.join('/tmp/',value))
                        
                        # Remaining attriburtes (bools, ints, doubles)
                        else:  
                            data[str(key)].append(value)

        logging.info(f'Converting data dict items to tensors...')
        for key, value in tqdm(data.items()):
            try:
                data[key] = torch.tensor(value)
            except ValueError:
                pass

        data['edge_index'] = edge_index.view(2, -1)
        data = torch_geometric.data.Data.from_dict(data)
        if data.x is None:
            data.num_nodes = G.number_of_nodes()


        logging.info('Selecting and concatenating features from data dict to data.x...')
        if self.valid_node_attrs is not None:
            xs = [data[key] for key in self.valid_node_attrs]
            xs = [x.view(-1, 1) if x.dim() <= 1 else x for x in xs]
            data.x = torch.cat(xs, dim=-1)
            for key in self.valid_node_attrs:
                data.__delattr__(key)

        logging.info('Selecting and concatenating features from data dict to data.img...')
        if self.img_node_attrs is not None:
            xs = [data[key] for key in self.img_node_attrs]
            xs = [x.view(-1, 1) if x.dim() <= 1 else x for x in xs]
            data.img = torch.cat(xs, dim=-1).permute(0, 3, 1, 2)
            for key in self.img_node_attrs:
                data.__delattr__(key)


        data.y = torch.tensor(list(nx.get_node_attributes(G, self.node_label).values()), dtype=torch.long)
        data.test_mask = torch.tensor(list(nx.get_node_attributes(G, 'test').values()), dtype=torch.bool)
        data.val_mask = torch.tensor(list(nx.get_node_attributes(G, 'val').values()), dtype=torch.bool)
        data.train_mask = torch.tensor(list(nx.get_node_attributes(G, 'train').values()), dtype=torch.bool)

        logging.info('Normalizing features...')
        for i in range(2,data.x.shape[1]):
            logging.info(f'feat{i:02d}: max: {data.x[:,i].max()}, min: {data.x[:,i].min()}, mean: {data.x[:,i].mean()}, std: {data.x[:,i].std()}')
            logging.info(f'normalized {i}-th feature:')
            data.x[:,i] = ( data.x[:,i] - data.x[:,i].mean() ) / data.x[:,i].std()
            logging.info(f'feat{i:02d}: max: {data.x[:,i].max()}, min: {data.x[:,i].min()}, mean: {data.x[:,i].mean()}, std: {data.x[:,i].std()}')

        # Convert data.img to
        data.img = data.img.float()

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        data.img = transform(data.img/255)

        data_list = [data]
            
        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 

class OSMChengduNoImage(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None):
        super(OSMChengduNoImage, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # List of the raw files
        return ['srn_processed.graphml']

    @property
    def processed_file_names(self):
        return 'osm_chengdu_no_image.pt'

    def download(self):
        pass

    def process(self):
        self.valid_node_attrs = [ 'bridge', 'oneway','centroid', 'bearing', 'length', 'translated_geometry']
        self.stringified_node_attrs = ['centroid','translated_geometry']
        self.node_label = 'highway'


        graph_path = os.path.join(self.raw_dir, 'srn_processed.graphml')
        G = nx.read_graphml(graph_path)

        logging.debug(f'Read graph from {graph_path}: {nx.info(G)}')

        G = nx.convert_node_labels_to_integers(G)
        G = G.to_directed() if not nx.is_directed(G) else G
        edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

        if G.number_of_nodes() > 0:
            node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
        else:
            node_attrs = {}

        if G.number_of_edges() > 0:
            edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
        else:
            edge_attrs = {}

        logging.debug(f'{len(node_attrs)} node attributes: {node_attrs}')
        logging.debug(f'{len(edge_attrs)} edge attributes: {edge_attrs}')

        data = defaultdict(list)
        
    
        for i, (n, feat_dict) in tqdm(enumerate(G.nodes(data=True)), total=G.number_of_nodes()):
            if set(feat_dict.keys()) != set(node_attrs):
                logging.warning(f'Mismatching attributes for node {n}: got {feat_dict.keys()} expected {node_attrs}')
                raise ValueError('Not all nodes contain the same attributes')

            for key, value in feat_dict.items():
                if key in self.valid_node_attrs:

                    # Stringified attributes (lists and tuples)
                    if key in self.stringified_node_attrs:
                        data[str(key)].append(ast.literal_eval(value))
                    
                    # Remaining attriburtes (bools, ints, doubles)
                    else:  
                        data[str(key)].append(value)

        logging.info(f'Converting data dict items to tensors...')
        for key, value in tqdm(data.items()):
            try:
                data[key] = torch.tensor(value)
            except ValueError:
                pass

        data['edge_index'] = edge_index.view(2, -1)
        data = torch_geometric.data.Data.from_dict(data)
        if data.x is None:
            data.num_nodes = G.number_of_nodes()


        logging.info('Selecting and concatenating features from data dict to data.x...')
        if self.valid_node_attrs is not None:
            xs = [data[key] for key in self.valid_node_attrs]
            xs = [x.view(-1, 1) if x.dim() <= 1 else x for x in xs]
            data.x = torch.cat(xs, dim=-1)
            for key in self.valid_node_attrs:
                data.__delattr__(key)


        data.y = torch.tensor(list(nx.get_node_attributes(G, self.node_label).values()), dtype=torch.long)
        data.test_mask = torch.tensor(list(nx.get_node_attributes(G, 'test').values()), dtype=torch.bool)
        data.val_mask = torch.tensor(list(nx.get_node_attributes(G, 'val').values()), dtype=torch.bool)
        data.train_mask = torch.tensor(list(nx.get_node_attributes(G, 'train').values()), dtype=torch.bool)

        logging.info('Normalizing features...')
        for i in range(2,data.x.shape[1]):
            logging.info(f'feat{i:02d}: max: {data.x[:,i].max()}, min: {data.x[:,i].min()}, mean: {data.x[:,i].mean()}, std: {data.x[:,i].std()}')
            logging.info(f'normalized {i}-th feature:')
            data.x[:,i] = ( data.x[:,i] - data.x[:,i].mean() ) / data.x[:,i].std()
            logging.info(f'feat{i:02d}: max: {data.x[:,i].max()}, min: {data.x[:,i].min()}, mean: {data.x[:,i].mean()}, std: {data.x[:,i].std()}')

        data_list = [data]

            
        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])  

class OSMChengduVFE(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, backbone=None, pretraining_dataset=None):
        # Setting the visual feature encoder (VFE) and processed file name
        if pretraining_dataset == 'ImageNet':
            if backbone == 'ResNet18':
                self.vfe = torchvision.models.resnet18(pretrained=True)
                self.processed_file_name = 'osm_chengdu_ResNet18_ImageNet.pt'
            elif backbone == 'ResNet50':
                self.vfe = torchvision.models.resnet50(pretrained=True)
                self.processed_file_name = 'osm_chengdu_ResNet50_ImageNet.pt'
        
        elif pretraining_dataset == 'NWPU-RESISC45':
            if backbone == 'ResNet18':
                self.vfe = torch.load('../pretrained_vfe/resnet18_on_NWPU-RESISC45_split_v2').to('cpu')
                self.processed_file_name = 'osm_chengdu_ResNet18_NWPU-RESISC45.pt'
            elif backbone == 'ResNet50':
                self.vfe = torch.load('../pretrained_vfe/resnet50_on_NWPU-RESISC45_split_v2').to('cpu')
                self.processed_file_name = 'osm_chengdu_ResNet50_NWPU-RESISC45.pt'
            
        for param in self.vfe.parameters():
                    param.requires_grad = False
        self.vfe = torch.nn.Sequential(*(list(self.vfe.children())[:-1]))
        self.vfe = torch.nn.Sequential(self.vfe, torch.nn.Flatten())

        super(OSMChengduVFE, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # List of the raw files
        return ['srn_processed.graphml', 'image_data.zip']

    @property
    def processed_file_names(self):
        return self.processed_file_name

    def download(self):
        pass

    def process(self):
        self.valid_node_attrs = ['bridge','oneway', 'oneway','centroid', 'bearing', 'length', 'translated_geometry']
        self.img_node_attrs = ['ortho']
        self.stringified_node_attrs = ['centroid','translated_geometry']
        self.node_label = 'highway'

        img_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        graph_path = os.path.join(self.raw_dir, 'srn_processed.graphml')
        G = nx.read_graphml(graph_path)

        logging.debug(f'Read graph from {graph_path}: {nx.info(G)}')

        G = nx.convert_node_labels_to_integers(G)
        G = G.to_directed() if not nx.is_directed(G) else G
        edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

        if G.number_of_nodes() > 0:
            node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
        else:
            node_attrs = {}

        if G.number_of_edges() > 0:
            edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
        else:
            edge_attrs = {}

        logging.debug(f'{len(node_attrs)} node attributes: {node_attrs}')
        logging.debug(f'{len(edge_attrs)} edge attributes: {edge_attrs}')

        data = defaultdict(list)
        zip_path = os.path.join(self.raw_dir, 'image_data.zip')
        with zipfile.ZipFile(zip_path, 'r') as myzip:
            logging.info(f'Converting node attributes to data dict...')
            for i, (n, feat_dict) in tqdm(enumerate(G.nodes(data=True)), total=G.number_of_nodes()):
                if set(feat_dict.keys()) != set(node_attrs):
                    logging.warning(f'Mismatching attributes for node {n}: got {feat_dict.keys()} expected {node_attrs}')
                    raise ValueError('Not all nodes contain the same attributes')

                for key, value in feat_dict.items():
                    if key in self.valid_node_attrs or key in self.img_node_attrs:

                        # Stringified attributes (lists and tuples)
                        if key in self.stringified_node_attrs:
                            data[str(key)].append(ast.literal_eval(value))
                        
                        # Image attributes 
                        elif key in self.img_node_attrs:
                            # Read corresponding image patch from zip file
                            tmp_arr = np.load(myzip.extract(value, path='/tmp')) 

                            if tmp_arr.dtype == np.uint16:
                                tmp_arr = tmp_arr.astype(np.int16)

                            data[str(key)].append(tmp_arr)
                            os.remove(os.path.join('/tmp/',value))
                        
                        # Remaining attribrutes (bools, ints, doubles)
                        else:  
                            data[str(key)].append(value)

        logging.info(f'Converting data dict items to tensors...')
        for key, value in tqdm(data.items()):
            try:
                data[key] = torch.tensor(value)
            except ValueError:
                pass

        data['edge_index'] = edge_index.view(2, -1)
        data = torch_geometric.data.Data.from_dict(data)
        if data.x is None:
            data.num_nodes = G.number_of_nodes()

        logging.info('Selecting and concatenating features from data dict to data.x...')
        if self.valid_node_attrs is not None:
            xs = [data[key] for key in self.valid_node_attrs]
            xs = [x.view(-1, 1) if x.dim() <= 1 else x for x in xs]
            data.x = torch.cat(xs, dim=-1)
            for key in self.valid_node_attrs:
                data.__delattr__(key)

        logging.info('Selecting and concatenating features from data dict to data.img...')
        if self.img_node_attrs is not None:
            xs = [data[key] for key in self.img_node_attrs]
            xs = [x.view(-1, 1) if x.dim() <= 1 else x for x in xs]
            data.img = torch.cat(xs, dim=-1).permute(0, 3, 1, 2)
            for key in self.img_node_attrs:
                data.__delattr__(key)


        data.y = torch.tensor(list(nx.get_node_attributes(G, self.node_label).values()), dtype=torch.long)
        data.test_mask = torch.tensor(list(nx.get_node_attributes(G, 'test').values()), dtype=torch.bool)
        data.val_mask = torch.tensor(list(nx.get_node_attributes(G, 'val').values()), dtype=torch.bool)
        data.train_mask = torch.tensor(list(nx.get_node_attributes(G, 'train').values()), dtype=torch.bool)

        logging.info('Normalizing features...')
        for i in range(2,data.x.shape[1]):
            logging.info(f'feat{i:02d}: max: {data.x[:,i].max()}, min: {data.x[:,i].min()}, mean: {data.x[:,i].mean()}, std: {data.x[:,i].std()}')
            logging.info(f'normalized {i}-th feature:')
            data.x[:,i] = ( data.x[:,i] - data.x[:,i].mean() ) / data.x[:,i].std()
            logging.info(f'feat{i:02d}: max: {data.x[:,i].max()}, min: {data.x[:,i].min()}, mean: {data.x[:,i].mean()}, std: {data.x[:,i].std()}')

        logging.info('Converting data.img to float...')
        data.img = data.img.float()

        logging.info('Transform data.img')
        data.img = img_transform(data.img/255)

        logging.info('Extract deep visual features')        
        data.img = self.vfe(data.img)

        data_list = [data]

        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 


class OSMChengduHist(InMemoryDataset):
    
    def __init__(self, root, transform=None, pre_transform=None, hist_bins=32):
        # Setting the visual feature encoder (VFE) and processed file name
        self.hist_bins = hist_bins
        super(OSMChengduHist, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # List of the raw files
        return ['srn_processed.graphml', 'image_data.zip']

    @property
    def processed_file_names(self):
        return 'osm_chengdu_hist.pt'

    def download(self):
        pass

    def process(self):
        self.valid_node_attrs = ['bridge','oneway', 'oneway','centroid', 'bearing', 'length', 'translated_geometry']
        self.img_node_attrs = ['ortho']
        self.stringified_node_attrs = ['centroid','translated_geometry']
        self.node_label = 'highway'

        img_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        graph_path = os.path.join(self.raw_dir, 'srn_processed.graphml')
        G = nx.read_graphml(graph_path)

        logging.debug(f'Read graph from {graph_path}: {nx.info(G)}')

        G = nx.convert_node_labels_to_integers(G)
        G = G.to_directed() if not nx.is_directed(G) else G
        edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

        if G.number_of_nodes() > 0:
            node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
        else:
            node_attrs = {}

        if G.number_of_edges() > 0:
            edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
        else:
            edge_attrs = {}

        logging.debug(f'{len(node_attrs)} node attributes: {node_attrs}')
        logging.debug(f'{len(edge_attrs)} edge attributes: {edge_attrs}')

        data = defaultdict(list)
        zip_path = os.path.join(self.raw_dir, 'image_data.zip')
        with zipfile.ZipFile(zip_path, 'r') as myzip:
            logging.info(f'Converting node attributes to data dict...')
            for i, (n, feat_dict) in tqdm(enumerate(G.nodes.data()), total=G.number_of_nodes()):
                if set(feat_dict.keys()) != set(node_attrs):
                    logging.warning(f'Mismatching attributes for node {n}: got {feat_dict.keys()} expected {node_attrs}')
                    raise ValueError('Not all nodes contain the same attributes')

                for key, value in feat_dict.items():
                    if key in self.valid_node_attrs or key in self.img_node_attrs:

                        # Stringified attributes (lists and tuples)
                        if key in self.stringified_node_attrs:
                            data[str(key)].append(ast.literal_eval(value))
                        
                        # Image attributes 
                        elif key in self.img_node_attrs:
                            # Read corresponding image patch from zip file
                            tmp_arr = np.load(myzip.extract(value, path='/tmp')) 

                            if tmp_arr.dtype == np.uint16:
                                tmp_arr = tmp_arr.astype(np.int16)
                        
                            data[str(key)].append(tmp_arr)

                            os.remove(os.path.join('/tmp/',value))
                        
                        # Remaining attribrutes (bools, ints, doubles)
                        else:  
                            data[str(key)].append(value)

        logging.info(f'Converting data dict items to tensors...')
        for key, value in tqdm(data.items()):
            try:
                data[key] = torch.tensor(value)
            except ValueError:
                pass

        data['edge_index'] = edge_index.view(2, -1)
        data = torch_geometric.data.Data.from_dict(data)
        if data.x is None:
            data.num_nodes = G.number_of_nodes()


        logging.info('Selecting and concatenating features from data dict to data.x...')
        if self.valid_node_attrs is not None:
            xs = [data[key] for key in self.valid_node_attrs]
            xs = [x.view(-1, 1) if x.dim() <= 1 else x for x in xs]
            data.x = torch.cat(xs, dim=-1)
            for key in self.valid_node_attrs:
                data.__delattr__(key)

        logging.info('Selecting and concatenating features from data dict to data.img...')
        if self.img_node_attrs is not None:
            xs = [data[key] for key in self.img_node_attrs]
            xs = [x.view(-1, 1) if x.dim() <= 1 else x for x in xs]
            data.img = torch.cat(xs, dim=-1)
            for key in self.img_node_attrs:
                data.__delattr__(key)


        data.y = torch.tensor(list(nx.get_node_attributes(G, self.node_label).values()), dtype=torch.long)
        data.test_mask = torch.tensor(list(nx.get_node_attributes(G, 'test').values()), dtype=torch.bool)
        data.val_mask = torch.tensor(list(nx.get_node_attributes(G, 'val').values()), dtype=torch.bool)
        data.train_mask = torch.tensor(list(nx.get_node_attributes(G, 'train').values()), dtype=torch.bool)

        logging.info('Normalizing features...')
        for i in range(2,data.x.shape[1]):
            logging.info(f'feat{i:02d}: max: {data.x[:,i].max()}, min: {data.x[:,i].min()}, mean: {data.x[:,i].mean()}, std: {data.x[:,i].std()}')
            logging.info(f'normalized {i}-th feature:')
            data.x[:,i] = ( data.x[:,i] - data.x[:,i].mean() ) / data.x[:,i].std()
            logging.info(f'feat{i:02d}: max: {data.x[:,i].max()}, min: {data.x[:,i].min()}, mean: {data.x[:,i].mean()}, std: {data.x[:,i].std()}')

        logging.info('Converting data.img to float...')
        data.img = data.img.float()

        logging.info('Extract histogram visual features')
        print('data.img:', data.img.shape)
        list_of_hists = []
        for x in data.img:
            print('x:', x.shape)
            x = x.permute(2,0,1)
            hists = []
            for channel_no, channel in enumerate(x):
                # print('channel:',channel_no, channel.shape)
                h = torch.histc(channel, bins=self.hist_bins, min=0, max=255)/(120*120)
                hists.append(h)
            # print('hists:', np.shape(hists))
            list_of_hists.append(torch.cat(hists))
        # print('list_of_hists:', np.shape(list_of_hists))
        data.img = torch.stack(list_of_hists)
        # print(data.img.shape)

        data_list = [data]

        # Apply the functions specified in pre_filter and pre_transform
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0]) 