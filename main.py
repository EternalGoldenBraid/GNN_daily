import torch.nn.functional as F
from event_tag_dataset import EventTagDataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np

from model import Model


ds = EventTagDataset()
ds.load_data(target='rating_sleep')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_channels = 2
hidden_channels = 6
out_channels = 5
model = Model(input_channels, hidden_channels, out_channels, device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

batch_size = 16
validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
indices = ds.data.x.index.values
split = int(np.floor(validation_split * len(indices)))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

train_x = ds.data.x.iloc[train_indices]
train_e = ds.data.edge_index
val_x = ds.data.x.iloc[val_indices]
val_e = ds.data.edge_index

train_x = torch.Tensor(train_x.astype(int).values)
val_x = torch.Tensor(val_x.astype(int).values)
train_e = torch.from_numpy(train_e.astype(int))
val_e = torch.from_numpy(val_e.astype(int))

#train_x = torch.Tensor(train_x.values)
#train_e = torch.from_numpy(train_e)

for epoch in range(200):
    pred = model(train_x, train_e)
    loss = F.cross_entropy(pred, val_e)

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
