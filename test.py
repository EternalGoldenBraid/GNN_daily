import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

#data = Data(x=x, edge_index=edge_index)
dataset = Planetoid('data','Cora')
data = dataset[0]
print(data)
