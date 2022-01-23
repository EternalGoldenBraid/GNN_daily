import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from model import ModelShallow, ModelDeep
from model_GAT import GATModel
from k_CV import k_fold
from train import train

from event_tag_dataset import EventTagDataset

### Hyperparameters

# Data
shuffle = False
target = 'rating_sleep'
#target = 'rating_day'
ds = EventTagDataset(shuffle=shuffle, target=target)
#batch_size=32
#loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

# Model
input_channels = (ds.data.x.shape[1])
hidden_channels = 32
out_channels = 5
save_model = False
model = 'GAT'
heads = 1
dropout = 0.2
# Optimizer
lr = 0.01
# Training
epochs = 2000
# K-fold validation
#k = 5

if model == 'GCN_shallow':
    model = ModelShallow(input_channels, hidden_channels, out_channels)
elif model == 'GCN_deep':
    model = ModelDeep(input_channels, hidden_channels, out_channels)
elif model == 'GAT':
    model = GATModel(input_channels, hidden_channels, out_channels, heads=heads, dropout=dropout)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#results = {}
#seed = 42
#torch.manual_seed(seed)
#train_indices, test_indices, val_indices = k_fold(ds, k, seed=seed)
#train_indices, test_indices= k_fold(ds, k, seed=seed)

# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
ds.data.to(device)

# Training
model.train()
train(model, ds, optimizer, epochs)

#model.eval()
#pred = model(ds.data.x, ds.data.edge_index).argmax(dim=1)
#correct = (pred[ds.data.test_mask] == ds.data.y[ds.data.test_mask]).sum()
#acc = int(correct) / int(ds.data.test_mask.sum())
#print(f'Accuracy: {acc:.4f}, {int(correct)}/{int(ds.data.test_mask.sum())}')

if save_model: model.save()
