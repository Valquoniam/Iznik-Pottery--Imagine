from fast_pytorch_kmeans import KMeans
import torch

device = 'cuda:0'
shape = (50000, 2)
cluster1 = torch.normal(mean=torch.zeros(shape), std=torch.ones(shape)).to(device)
cluster2 = torch.normal(mean=torch.full(shape, 2.0), std=torch.ones(shape)).to(device)
cluster3 = torch.normal(mean=torch.full(shape, -3.0), std=torch.ones(shape)).to(device)

x = torch.cat([cluster1, cluster2, cluster3], dim=0)

kmeans = KMeans(n_clusters=3, mode='euclidean', verbose=1)
labels = kmeans.fit_predict(x)
print('hi')