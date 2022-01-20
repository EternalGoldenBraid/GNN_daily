import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from model import ModelShallow, ModelDeep
from k_CV import k_fold
from train import train

from event_tag_dataset import EventTagDataset

ds = EventTagDataset()
#batch_size=32
#loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

# Model
input_channels = 2
hidden_channels = 32
out_channels = 6
save_model = False
model = 'shallow'
if model == 'shallow':
    model = ModelShallow(input_channels, hidden_channels, out_channels)
else:
    model = ModelDeep(input_channels, hidden_channels, out_channels)

lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# K-fold validation
#k = 5
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
epochs = 200
model.train()
train(model, ds, optimizer, epochs)

#model.eval()
#pred = model(ds.data.x, ds.data.edge_index).argmax(dim=1)
#correct = (pred[ds.data.test_mask] == ds.data.y[ds.data.test_mask]).sum()
#acc = int(correct) / int(ds.data.test_mask.sum())
#print(f'Accuracy: {acc:.4f}, {int(correct)}/{int(ds.data.test_mask.sum())}')

if save_model: model.save()
