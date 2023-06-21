import torch

class DimRed:
    def __init__(self, n_components=2, device='cuda'):
        self.n_components = n_components
        k = torch.rand(256, )
        k /= k.norm()
        u = torch.rand(256, )
        u -= u.dot(k) * k / k.norm() ** 2
        self.components = torch.stack([k, u], dim=0).T.to(device)

    def fit_transform(self, X):
        dimred = X @ self.components
        return dimred
