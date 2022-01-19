import torch
import pandas as pd
import numpy as np
#from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Dataset, Data
#from sklearn.model_selection import train_test_split
#import torch_geometric.transforms as T

import os
import pickle

# custom dataset
class EventTagDataset(Dataset):
    def __init__(self, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.data = Data()
        self.dates = None
        self.size =  None
        self.target = 'rating_sleep'

        # Load data
        # features: pandas.df, Edges: np.array
        with open('data/event_tag_graph.dat', 'rb') as f:
            features, edges = pickle.load(f)

        self.dates = features['date']
        features.drop('date', axis=1, inplace=True)

        self.data.y = torch.Tensor(features[self.target].values).long()
        features.drop(self.target, axis=1, inplace=True)

        self.data.edge_index = torch.from_numpy(edges.T)
        self.data.x = torch.Tensor(features.values)

        self.size = features.shape[0]

        perm = torch.randperm(self.size)

        split = [0.7, 0.1, 0.2]
        train_idx = int(np.floor(split[0]*self.size))
        val_idx = int(np.floor(split[1]*self.size))

        train_mask = torch.zeros(self.size, dtype=torch.long)
        train_mask[perm[0:train_idx]] = 1
        self.data.train_mask = train_mask

        val_mask = torch.zeros(self.size, dtype=torch.long)
        val_mask[perm[train_idx:val_idx]] = 1
        self.data.val_mask = val_mask

        test_mask = torch.zeros(self.size, dtype=torch.long)
        test_mask[perm[val_idx:-1]] = 1
        self.data.test_mask = test_mask

        self.data.num_node_features = features.shape[1]


    def _download(self):
        return

    def load_data(self, target='rating_sleep'):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
