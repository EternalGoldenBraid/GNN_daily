import torch
import pandas as pd
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
       self.data = None
       self.dates = None
       self.size =  None

    def _download(self):
        return

    def load_data(self, target='rating_sleep'):
        # Read data into data into `Data` list.
        with open('data/event_tag_graph.dat', 'rb') as f:
            features, edges = pickle.load(f)
        data = Data()
        self.dates = features['date']
        features.drop('date', axis=1, inplace=True)
        data.y = features[target]
        features.drop(target, axis=1, inplace=True)
        data.edge_index = edges.T
        data.x = features
        self.data = data
        self.size = features.shape[0]
        return

    def _process(self):
        return

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
