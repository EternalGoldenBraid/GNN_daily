from sklearn.model_selection import StratifiedKFold
import torch
import pickle
import os

def k_fold(dataset, folds, seed):
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)

    test_indices, train_indices = [], []
    print(dataset.data.y.shape)
    print(torch.zeros((dataset.size)))
    for _, idx in skf.split(torch.zeros((dataset.size)), dataset.data.y.reshape(-1,1)):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    #val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(dataset.size, dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        #train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
   
    #return train_indices, test_indices, val_indices


    #THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    #with open(
    return train_indices, test_indices
