import torch.nn.functional as F
import torch
from sklearn.metrics import confusion_matrix
from callbacks import CallbackHandler
import matplotlib.pyplot as plt

#def train(model, ds, optimizer, train_indices, val_indices, epochs):
def train(model, ds, optimizer, epochs, callback=CallbackHandler):

    # Train
    freq = 30
    predictions = torch.empty(epochs)
    for epoch in range(epochs):

        model.train()
        pred = model(ds.data.x, ds.data.edge_index)
        loss = F.cross_entropy(pred[ds.data.train_mask], ds.data.y[ds.data.train_mask].long())
        #loss = F.nll_loss(pred[ds.data.train_mask], ds.data.y[ds.data.train_mask])
    
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validate
        model.eval()
        pred = model(ds.data.x, ds.data.edge_index).argmax(dim=1)

        if epoch % freq == 0:
            print("epoch:", epoch)
            print("loss:", loss.item())
            callback.accuracy(pred[ds.data.test_mask], ds.data.y[ds.data.test_mask])

        if epoch % int(epochs/3) == 0:
            m = callback.cmatrix(pred[ds.data.test_mask].cpu(), ds.data.y[ds.data.test_mask].cpu())
            #m = callback.cmatrix(pred[ds.data.test_mask].copy(), ds.data.y[ds.data.test_mask].copy())
            #plt.imshow(m)
            plt.show()
        #predictions.append(pred)

