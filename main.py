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
loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
#for batch in loader:
#    print(batch)
#    print(batch.num_graphs)
#input()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = ds.data.y.float().mean()
std= ds.data.y.float().std()
plot = True
if plot == True:
    plt.plot(ds.data.y)
    plt.title(f"Mean: {mean}, std: {std}.")
    plt.show()
input_channels = 2
hidden_channels = 6
out_channels = 5
model = Model(input_channels, hidden_channels, out_channels, device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#loss = F.nll_loss(pred[ds.data.train_mask], ds.data.y[ds.data.train_mask])

for epoch in range(200):
    pred = model(ds.data.x, ds.data.edge_index)
    #loss = F.cross_entropy(pred, val_e)
    loss = F.cross_entropy(pred[ds.data.train_mask], ds.data.y[ds.data.train_mask].long())

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
pred = model(ds.data.x, ds.data.edge_index).argmax(dim=1)
correct = (pred[ds.data.test_mask] == ds.data.y[ds.data.test_mask]).sum()
acc = int(correct) / int(ds.data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')

print("Correct: ", int(correct)) 
print(int(ds.data.test_mask.sum()))
#plt.plot(pred)
#plt.show()

print(pred)

