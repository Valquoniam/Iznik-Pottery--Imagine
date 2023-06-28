import os
from PIL import Image
import matplotlib.pyplot as plt

# visualize tiles at different dimensions

dataroot = 'data/Iznik_tiles/'
sizes = (16, 24, 32, 48, 54, 64, 72, 96)
n_sizes = len(sizes)
n_show = 8
fontsize = 48

f, ax = plt.subplots(n_show, n_sizes, figsize=(n_sizes * 5, n_show * 5))
for i, fn in enumerate(os.listdir(dataroot)[:n_show]):
    base_img = Image.open(os.path.join(dataroot, fn))
    imgs = [base_img.resize((size, size)) for size in sizes]
    for j, img in enumerate(imgs):
        ax[i][j].imshow(img)
        ax[i][j].set_title(f'{sizes[j]}x{sizes[j]}', fontsize=fontsize)
        ax[i][j].axis('off')
    plt.tight_layout()
    plt.subplots_adjust(wspace=-0.2, hspace=0.3)
plt.show()