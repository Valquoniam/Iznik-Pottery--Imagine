import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from train_diffusion import GenModel
from utils.util import reverse_z_center, scale
from utils.display import show_images
import torchvision.transforms as T
import torchvision.utils as vutils
from tqdm import tqdm
from kmeans_pytorch import kmeans
from utils.display import remove_axes


def make_gif(frames, name):
    frame_one = frames[0]
    frame_one.save(f'gifs/{name}.gif', format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

if __name__ == '__main__':
    device = 'cuda:0'
    ckpt_path = './logs/tiles_orig_unet_size_48_steps_1000_lr_0.001/lightning_logs/version_2/checkpoints/last.ckpt'
    model = GenModel.load_from_checkpoint(ckpt_path).to(device)
    model.eval()

    n = 20000
    size = 48
    c = 3
    clusters = 5

    timesteps = list(range(1000))[::-1]
    t = timesteps[0]
    sample = torch.randn(n, c, size, size).to(device)

    time_tensor = (torch.ones(n, 1) * t).long().to(device)

    features, t, r, h = model.ddpm.network.extract_features(sample, time_tensor)

    x_flat = features.reshape((len(features), -1))
    data_size, dims = x_flat.shape

    n_show = 8

    with torch.no_grad():
        cluster_ids_x, cluster_centers = kmeans(
            X=x_flat, num_clusters=clusters, distance='euclidean', device=device)


        frames = []
        for i, t in enumerate(tqdm(timesteps)):
            time_tensor = (torch.ones(n, 1) * t).long().to(device)
            residual = model.ddpm.reverse(sample, time_tensor)
            sample = model.ddpm.step(residual, time_tensor[0], sample)

        for i in range(n):
            frames.append(sample[i].detach().cpu())

        frames = torch.stack(frames, dim=0)

    f, ax = plt.subplots(clusters, n_show, figsize=(n_show * 3, clusters * 3))

    for cluster_i in range(clusters):
        sample = frames[cluster_ids_x == cluster_i]
        cluster_subset = sample[:n_show]
        for j in range(n_show):
            frame = reverse_z_center(sample[j])
            ax[cluster_i][j].imshow(frame.permute(1, 2, 0).detach().cpu())
        ax[cluster_i][0].set_title(f'Cluster {cluster_i}', fontsize=24)
    remove_axes(ax)
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0.03, right=1, top=0.95, wspace=-0.5, hspace=0.3)
    plt.show()
    print()
