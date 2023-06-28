from cuml.common.device_selection import set_global_device_type
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler

from cuml import PCA as cuPCA
from sklearn.decomposition import PCA as skPCA

data = torch.randn((10000, 2)).cuda()
data[:, 0] *= 10

n_comp = 2
plt.scatter(data[:, 0].cpu(), data[:, 1].cpu(), label='data', s=1)
plt.gca().set_xlim(-50, 50)
plt.gca().set_ylim(-50, 50)
plt.legend()
plt.show()

set_global_device_type('CPU')
dimred_cpu = cuPCA(n_components=n_comp, svd_solver='full', whiten=False)
data_dimred = dimred_cpu.fit_transform(data)
plt.scatter(data_dimred[:, 0], data_dimred[:, 1], label='cpu', s=1)
plt.gca().set_xlim(-50, 50)
plt.gca().set_ylim(-50, 50)
plt.legend()
plt.show()

set_global_device_type('GPU')
dimred_gpu = cuPCA(n_components=n_comp, svd_solver='full', whiten=False)
dimred_gpu.fit(data)
data_dimred = dimred_gpu.fit_transform(data).get()
plt.scatter(data_dimred[:, 0], data_dimred[:, 1], label='gpu', s=1)
plt.gca().set_xlim(-50, 50)
plt.gca().set_ylim(-50, 50)
plt.legend()
plt.show()

print()
print(dimred_cpu.components_)
print(dimred_gpu.components_)