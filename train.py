import torch.nn.functional as F

#def train(model, ds, optimizer, train_indices, val_indices, epochs):
def train(model, ds, optimizer, epochs):

    # Train
    for epoch in range(epochs):
        pred = model(ds.data.x, ds.data.edge_index)
        loss = F.cross_entropy(pred[ds.data.train_mask], ds.data.y[ds.data.train_mask].long())
        #loss = F.nll_loss(pred[ds.data.train_mask], ds.data.y[ds.data.train_mask])
        print("epoch:", epoch)
        print("loss:", loss.item())
    
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validate
        model.eval()
        pred = model(ds.data.x, ds.data.edge_index).argmax(dim=1)
        correct = (pred[ds.data.test_mask] == ds.data.y[ds.data.test_mask]).sum()
        acc = int(correct) / int(ds.data.test_mask.sum())
        print(f'accuracy: {acc:.4f}')

