import matplotlib.pyplot as plt
from event_tag_dataset import EventTagDataset
from scipy import stats
import torch

### Hyperparameters

# Data
shuffle = False
#target = 'rating_sleep'
target = 'rating_day'
ds = EventTagDataset(shuffle=shuffle, target=target)
features = ds.data.x
target = ds.data.y

#print(stats.describe(target))
#print(stats.describe(features,axis=1))
print(features[:2])
print(features.type(torch.float).mean(dim=0))
print(features.type(torch.float).std(dim=0))

t = features[:,1]
#plt.scatter(list(range(len(t))), t.flatten())
#plt.plot(target)
#plt.show()
