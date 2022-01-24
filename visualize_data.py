import matplotlib.pyplot as plt
from event_tag_dataset import EventTagDataset
from scipy import stats
import numpy as np
import torch

### Hyperparameters

# Data
shuffle = False
#target = 'rating_sleep'
target = 'rating_day'
ds = EventTagDataset(shuffle=shuffle, target=target)
features = ds.feature_df

#print(stats.describe(target))
#print(stats.describe(features,axis=1))
#print(features[:2])
#print(features.type(torch.float).mean(dim=0))
#print(features.type(torch.float).std(dim=0))

print("Features: ")
print(features)

# Rating histograms
#features.hist(column=target)
features.hist()

# Tag_id histograms
#t = features[:,1].numpy()
#plt.hist(t,bins=np.unique(t))
#plt.plot(t, 'bo')

plt.show()
