import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch_lightning import seed_everything

from train_diffusion import GenModel
from utils.util import reverse_z_center, scale
from utils.display import show_images
import torchvision.transforms as T
import torchvision.utils as vutils
from tqdm import tqdm
from fast_pytorch_kmeans import KMeans
from utils.display import remove_axes
from sklearn.decomposition import PCA
from cuml.manifold import TSNE


def make_gif(frames, name):
    frame_one = frames[0]
    frame_one.save(f'gifs/{name}.gif', format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

if __name__ == '__main__':
    seed = 4321
    seed_everything(seed)
    device = 'cuda:0'

    # ckpt_path = './logs/tiles_orig_unet_size_48_steps_1000_lr_0.001/lightning_logs/version_2/checkpoints/last.ckpt'
    # size = 48

    ckpt_path = './logs/tiles_orig_unet_size_64_steps_1000_lr_0.001_aug_flip,rotate,symmetry/lightning_logs/version_0/checkpoints/epoch=119.ckpt'
    size = 64

    model = GenModel.load_from_checkpoint(ckpt_path).to(device)
    model.eval()

    c = 3
    n_show = 10

    n = 100000

    batch_size = 1000
    epochs = n // batch_size

    clusters = 10

    with torch.no_grad():
        timesteps = list(range(1000))[::-1]
        for timestep_i in [0]:  # , 200, 400, 600, 800, 999]:
            t = timesteps[timestep_i]

            features = []
            samples = []

            for i in tqdm(range(epochs)):
                batch_sample = torch.randn(batch_size, c, size, size).to(device)
                batch_time_tensor = (torch.ones(1, batch_size).to(device) * t).long().squeeze()
                batch_features, _, _, _ = model.ddpm.network.extract_features(batch_sample, batch_time_tensor)
                features.append(batch_features.mean((-1, -2)))
                samples.append(batch_sample)

            features = torch.cat(features, dim=0)
            samples = torch.cat(samples, dim=0)

            x_flat = features.reshape((len(features), -1))
            data_size, dims = x_flat.shape

            kmeans = KMeans(n_clusters=clusters, mode='euclidean', verbose=1)
            labels = kmeans.fit_predict(x_flat)


            samples_to_show = []
            for cluster_i in range(clusters):
                sample = samples[labels == cluster_i]
                samples_to_show.append(sample[:n_show])
            samples_to_show_cat = torch.cat(samples_to_show)
            n_show_total = len(samples_to_show_cat)

            for i, t_i in enumerate(tqdm(timesteps)):
                time_tensor = (torch.ones(n_show_total, 1) * t_i).long().to(device)
                residual = model.ddpm.reverse(samples_to_show_cat, time_tensor)
                samples_to_show_cat = model.ddpm.step(residual, time_tensor[0], samples_to_show_cat)
            frames = samples_to_show_cat.split(split_size=n_show, dim=0)

            f, ax = plt.subplots(clusters, n_show, figsize=(n_show * 1.5, clusters * 1.5))

            for cluster_i in range(clusters):
                cluster_subset = frames[cluster_i]
                for j in range(n_show):
                    frame = reverse_z_center(cluster_subset[j])
                    ax[cluster_i][j].imshow(frame.permute(1, 2, 0).detach().cpu())
                ax[cluster_i][0].set_title(f'Cluster {cluster_i}', fontsize=16)
            remove_axes(ax)
            plt.tight_layout()
            plt.subplots_adjust(left=0, bottom=0.03, right=1, top=0.95, wspace=-0.65, hspace=0.3)
            plt.show()

            # dimred = PCA(n_components=2)
            perplexity = 50
            dimred = TSNE(n_components=2, perplexity=perplexity, n_neighbors=perplexity * 5,
                          verbose=6, method='barnes_hut')
            feature_dimred = dimred.fit_transform(features)

            for cluster_i in range(clusters):
                feature_dimred_subset = feature_dimred.get()[labels.cpu() == cluster_i]
                plt.scatter(feature_dimred_subset[:, 0], feature_dimred_subset[:, 1], s=0.01)
            plt.title(f't-SNE ({clusters} clusters) @ t={t}')
            plt.tight_layout()
            plt.show()
    print('here')