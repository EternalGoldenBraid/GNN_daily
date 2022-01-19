import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from model import Model

from event_tag_dataset import EventTagDataset


ds = EventTagDataset()
batch_size=32
#loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
#for batch in loader:
#    print(batch)
#    print(batch.num_graphs)
#input()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cur_d = torch.cuda.current_device
print("Current d: ", torch.cuda.get_device_name(cur_d))

input_channels = 2
hidden_channels = 6
out_channels = 5
model = Model(input_channels, hidden_channels, out_channels, device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


epochs = 200
losses = np.zeros(epochs)
for epoch in range(epochs):
    #print("epoch: ", epoch)
    pred = model(ds.data.x, ds.data.edge_index)
    loss = F.cross_entropy(pred[ds.data.train_mask], ds.data.y[ds.data.train_mask].long())
    #loss = F.nll_loss(pred[ds.data.train_mask], ds.data.y[ds.data.train_mask])
    losses[epoch] = loss

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


model.eval()
pred = model(ds.data.x, ds.data.edge_index).argmax(dim=1)
correct = (pred[ds.data.test_mask] == ds.data.y[ds.data.test_mask]).sum()
acc = int(correct) / int(ds.data.test_mask.sum())
print(f'Accuracy: {acc:.4f}, {int(correct)}/{int(ds.data.test_mask.sum())}')
