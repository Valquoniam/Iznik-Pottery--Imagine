import os
from torch.utils.data import Dataset
from PIL import Image

class TilesDataset(Dataset):
    def __init__(self, dataroot, transform=None):
        self.transform = transform
        self.tiles = []
        for fn in os.listdir(dataroot):
            if fn.endswith('.jpeg'):
                path = os.path.join(dataroot, fn)
                self.tiles.append(path)
    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        path = self.tiles[idx]
        tile = Image.open(path)
        if self.transform:
            tile = self.transform(tile)
        return tile
