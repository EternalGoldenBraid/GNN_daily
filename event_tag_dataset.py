import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset, Data, InMemoryDataset
#import torch_geometric.transforms as T

import os
import pickle

# custom dataset
#class EventTagDataset(Dataset):
class EventTagDataset(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None,
            shuffle=False, target='rating_day'):
        super().__init__(None, transform, pre_transform)
        self.data = Data()
        self.dates = None
        self.size =  None
        self.target = target
        self.shuffle = shuffle
        self.feature_df = None

        # Load data
        # features: pandas.df, Edges: np.array
        # with open('data/event_tag_graph.dat', 'rb') as f:
        with open('event_tag_graph.dat', 'rb') as f:
            features, edges, cross_edges = pickle.load(f)

        self.feature_df = features.copy()
        if features['rating_sleep'].min() < 0:
            features['rating_sleep'] += np.abs(features['rating_sleep'].min())
        if features['rating_day'].min() < 0:
            features['rating_day'] += np.abs(features['rating_day'].min())

        #self.dates = features['date']
        #self.dates = dates
        #features.drop('date', axis=1, inplace=True)

        self.data.y = torch.Tensor(features[self.target].values).long()
        features.drop(self.target, axis=1, inplace=True)

        self.data.edge_index = torch.from_numpy(np.concatenate((edges.T, cross_edges.T),1))
        #self.data.x = torch.Tensor(features.values).type(torch.long)
        self.data.x = torch.Tensor(features.values)

        self.size = features.shape[0]

        if shuffle:
            perm = torch.randperm(self.size)
        else:
            perm = list(range(self.size))

        cross_val = False
        if cross_val:
            split = [0.7, 0.1, 0.2]
            train_idx = int(np.floor(split[0]*self.size))
            val_idx = train_idx + int(np.floor(split[1]*self.size))

            train_mask = torch.zeros(self.size, dtype=torch.bool)
            train_mask[perm[0:train_idx]] = 1
            self.data.train_mask = train_mask

            val_mask = torch.zeros(self.size, dtype=torch.bool)
            val_mask[perm[train_idx:val_idx]] = 1
            self.data.val_mask = val_mask

            test_mask = torch.zeros(self.size, dtype=torch.bool)
            test_mask[perm[val_idx:]] = 1
            self.data.test_mask = test_mask
        else:
            split = 0.7
            train_idx = int(np.floor(split*self.size))

            train_mask = torch.zeros(self.size, dtype=torch.bool)
            train_mask[perm[0:train_idx]] = 1
            self.data.train_mask = train_mask

            test_mask = torch.zeros(self.size, dtype=torch.bool)
            test_mask[perm[train_idx:]] = 1
            self.data.test_mask = test_mask

        self.data.num_node_features = features.shape[1]

    def _download(self):
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
