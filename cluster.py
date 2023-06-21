import matplotlib.pyplot as plt
import numpy as np
import torch

from fast_pytorch_kmeans import KMeans
from pytorch_lightning import seed_everything
from tqdm import tqdm
import os

from train_diffusion import GenModel
from utils.display import remove_axes, remove_spines
from utils.util import reverse_z_center
from utils.dimred import DimRed


def nearest_neighbor(cloud, center):
    center = center.expand(cloud.shape)
    dist = (cloud - center).pow(2).sum(dim=1).pow(.5)
    knn_indices = dist.topk(2, largest=False, sorted=True)[1][1]
    return cloud[knn_indices]


def make_gif(frames, name):
    frame_one = frames[0]
    frame_one.save(f'gifs/{name}.gif', format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)


def get_quadrant(point, scatterplot):
    if dimred_device == 'cpu':
        return torch.randint(low=0, high=4, size=()).item()
    else:
        x, y = point
        x_limits = scatterplot.get_xlim()
        y_limits = scatterplot.get_ylim()

        x_center = (x_limits[1] + x_limits[0]) / 2
        y_center = (y_limits[1] + y_limits[0]) / 2

        if x >= x_center:
            if y >= y_center:
                return 1
            else:
                return 4
        else:
            if y >= y_center:
                return 2
            else:
                return 3


def generate(n_show_total, timesteps, samples_to_show_cat):
    for i, t_i in enumerate(tqdm(timesteps)):
        time_tensor = (torch.ones(n_show_total, 1) * t_i).long().to(device)
        residual = model.ddpm.reverse(samples_to_show_cat, time_tensor)
        samples_to_show_cat = model.ddpm.step(residual, time_tensor[0], samples_to_show_cat)
    return samples_to_show_cat

def generate_from_features(features, t, r, h):
    for i, t_i in enumerate(tqdm(timesteps)):
        time_tensor = (torch.ones(n_show_total, 1) * t_i).long().to(device)
        residual = model.ddpm.upsample(features, t, r, h)
        samples = model.ddpm.step(residual, time_tensor[0], samples)
    return samples

def set_img_border(img, border_color, border_size):
    for side in ['bottom', 'top', 'left', 'right']:
        img.spines[side].set_color(border_color)
        img.spines[side].set_linewidth(border_size)


if __name__ == '__main__':
    seed = 4321
    seed_everything(seed)
    device = 'cuda'
    dimred_device = 'gpu'

    if dimred_device == 'cpu':
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
    else:
        from cuml.manifold import TSNE
        from cuml import PCA

    c = 3
    size = 64
    ckpt_root = './logs/ddpm_unet/'

    # ckpt_path = 'tiles_orig_unet_size_64_steps_1000_lr_0.001/' \
    #             'lightning_logs/version_0/checkpoints/last.ckpt'

    # ckpt_path = 'tiles_orig_unet_size_64_steps_1000_lr_0.001_aug_flip,rotate,symmetry/' \
    #             'lightning_logs/version_0/checkpoints/epoch=956.ckpt'

    # ckpt_path = 'tiles_orig_unet_size_64_steps_1000_lr_0.001_aug_flip/' \
    #             'lightning_logs/version_0/checkpoints/epoch=2682.ckpt'

    # ckpt_path = 'tiles_orig_unet_size_64_steps_1000_lr_0.001_aug_rotate/' \
    #             'lightning_logs/version_0/checkpoints/epoch=7131.ckpt'

    ckpt_path = 'tiles_orig_unet_size_64_steps_1000_lr_0.001_aug_symmetry/' \
                'lightning_logs/version_0/checkpoints/epoch=4679.ckpt'

    ckpt_path = os.path.join(ckpt_root, ckpt_path)
    model = GenModel.load_from_checkpoint(ckpt_path).to(device)
    model.eval()

    # ------------ Hyperparameters to change
    n = 50000
    clusters = 10
    timesteps_to_cluster = [0]  # , 200, 400, 600, 800, 999]

    dimred_mode = 'pca' # 'tsne' or 'pca' or 'orth_dimred'
    dimred_space = 'features'  # 'features' or 'imgs'

    cluster_mode = 'points_imgpairs' # 'points_imgrows' or 'points_imgpairs' or 'steer' or 'imgs'
    n_show = 10
    n_imgs_show = 100
    cluster_dim = 'full'  # 'dimred' or 'full'
    # ------------ Hyperparameters to change

    scatter_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                      '#17becf', '#008080', '#000000', '#FFFF33', '#33FFFF', '#00FF55']
    batch_size = 500
    lw = 5
    epochs = n // batch_size

    with torch.no_grad():
        timesteps = list(range(1000))[::-1]
        for timestep_i in timesteps_to_cluster:
            t = timesteps[timestep_i]

            features = []
            samples = []

            for i in tqdm(range(epochs)):
                batch_sample = torch.randn(batch_size, c, size, size).to(device)
                batch_time_tensor = (torch.ones(1, batch_size).to(device) * t).long().squeeze()
                batch_features, t_emb, r, h = model.ddpm.network.extract_features(batch_sample, batch_time_tensor)
                features.append(batch_features.mean((-1, -2)))
                samples.append(batch_sample)

            features = torch.cat(features, dim=0)
            samples = torch.cat(samples, dim=0)

            if dimred_mode == 'tsne':
                perplexity = 50
                dimred = TSNE(n_components=2, perplexity=perplexity, n_neighbors=perplexity * 5,
                              method='barnes_hut')
            elif dimred_mode == 'pca':
                dimred = PCA(n_components=2, svd_solver='full', whiten=True)
            elif dimred_mode == 'orth_dimred':
                dimred = DimRed(n_components=2, device=device)

            if dimred_space == 'imgs':
                data = samples
                data_dimred = torch.tensor(dimred.fit_transform(samples.reshape((len(samples), -1))))
            elif dimred_space == 'features':
                data = features
                if dimred_device == 'cpu':
                    data = data.cpu()
                data_dimred = torch.tensor(dimred.fit_transform(data))
            data_dimred = data_dimred.cpu()

            if cluster_dim == 'full':
                x_flat = data.reshape((len(data_dimred), -1))
            elif cluster_dim == 'orth_dimred':
                x_flat = data_dimred.reshape((len(data_dimred), -1))
            data_size, dims = x_flat.shape

            kmeans = KMeans(n_clusters=clusters, mode='euclidean', verbose=1)
            labels = kmeans.fit_predict(x_flat)

            if cluster_mode == 'points_imgrows':
                for cluster_i in range(clusters):
                    data_dimred_subset = data_dimred[labels.cpu() == cluster_i]
                    plt.scatter(data_dimred_subset[:, 0], data_dimred_subset[:, 1], s=0.01, alpha=1.0)
                plt.title(f't-SNE ({clusters} clusters) @ t={t}')
                plt.tight_layout()
                plt.show()

                samples_to_show = []
                for cluster_i in range(clusters):
                    sample = samples[labels == cluster_i]
                    samples_to_show.append(sample[:n_show])
                samples_to_show_cat = torch.cat(samples_to_show)
                n_show_total = len(samples_to_show_cat)

                frames = generate(n_show_total, timesteps, samples_to_show_cat)
                frames = frames.split(split_size=n_show, dim=0)

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

            elif cluster_mode == 'points_imgpairs':
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))

                sc_pad = 0.2
                sc = ax.inset_axes([sc_pad, sc_pad, 1 - (2 * sc_pad), 1 - (2 * sc_pad)], zorder=-1)

                for cluster_i in range(clusters):
                    data_dimred_subset = data_dimred[labels.cpu() == cluster_i]
                    sc.scatter(data_dimred_subset[:, 0], data_dimred_subset[:, 1], s=0.03, alpha=0.8,
                               c=scatter_colors[cluster_i % len(scatter_colors)])

                img_s = 0.18
                img_padding = (sc_pad - img_s) / 2
                n_imgs = (1 / sc_pad) * 4 - 4
                n_imgs_per_quadrant = int(n_imgs // 4)

                quadrants = [[], [], [], []]
                start_x = 0.6
                start_y = 0
                for q_i, quadrant in enumerate(quadrants):
                    for i in range(n_imgs_per_quadrant):
                        q_ax = ax.inset_axes([start_x + img_padding, start_y + img_padding, img_s, img_s])
                        remove_axes(q_ax)
                        quadrant.append(q_ax)

                        if q_i == 0:
                            if start_x + sc_pad < 1:
                                start_x += sc_pad
                            else:
                                start_y += sc_pad
                        elif q_i == 1:
                            if start_y + sc_pad < 1:
                                start_y += sc_pad
                            else:
                                start_x -= sc_pad
                        elif q_i == 2:
                            if start_x - sc_pad > 0:
                                start_x -= sc_pad
                            else:
                                start_y -= sc_pad
                        else:
                            if start_y - sc_pad > 0:
                                start_y -= sc_pad
                            else:
                                start_x += sc_pad

                remove_axes(ax)
                remove_axes(sc)

                remove_spines(ax)
                remove_spines(sc)

                sorted_x, indices = torch.sort(data_dimred, dim=0, descending=False)
                data_dimred_sorted_x_total = torch.stack([sorted_x[:, 0], data_dimred[:, 1][indices[:, 0]]],
                                                            dim=1)
                labels_sorted_x = labels[indices[:, 0]]
                sorted_y, indices = torch.sort(data_dimred, dim=0, descending=False)
                data_dimred_sorted_y_total = torch.stack([data_dimred[:, 0][indices[:, 1]], sorted_x[:, 1]],
                                                            dim=1)
                labels_sorted_y = labels[indices[:, 1]]
                data_dimred_sorted = [[data_dimred_sorted_x_total, labels_sorted_x, [0 for _ in range(clusters)]],
                                         [data_dimred_sorted_y_total, labels_sorted_y, [0 for _ in range(clusters)]]]
                from_start = True
                sorted_idx = 0

                quadrant_count = [0, 0, 0, 0]
                quadrant_order = []
                image_pairs = []
                coords = []
                cluster_order = []
                clusters_left = list(range(clusters))
                while not all([q == 2 for q in quadrant_count]):
                    cluster_i = np.random.choice(clusters_left)

                    features_subset = features[labels.cpu() == cluster_i]
                    feature_dimred_subset = data_dimred[labels.cpu() == cluster_i]
                    sorted_to_use, labels_sorted_to_use, idx_to_use = data_dimred_sorted[sorted_idx]
                    idx_to_use = idx_to_use[cluster_i]
                    sorted_to_use_subset = sorted_to_use[labels_sorted_to_use.cpu() == cluster_i]
                    idx_to_use_diff = len(sorted_to_use_subset) // 2 + idx_to_use
                    if from_start:
                        rand_point = sorted_to_use_subset[idx_to_use_diff]
                    else:
                        rand_point = sorted_to_use_subset[-idx_to_use_diff]
                    data_dimred_sorted[sorted_idx][2][cluster_i] += 1
                    sorted_idx = 1 - sorted_idx
                    from_start = not from_start

                    rand_idcs = torch.all(data_dimred == rand_point, dim=1).nonzero().item()

                    if cluster_dim == 'full':
                        rand_point = features[rand_idcs]
                        nn_point = nearest_neighbor(features_subset, rand_point)

                        rand_idcs = torch.all(features == rand_point, dim=1).nonzero().item()
                        nn_idcs = torch.all(features == nn_point, dim=1).nonzero().item()
                    elif cluster_dim == 'dimred':
                        rand_point = data_dimred[rand_idcs]
                        nn_point = nearest_neighbor(feature_dimred_subset, rand_point)

                        rand_idcs = torch.all(data_dimred == rand_point, dim=1).nonzero().item()
                        nn_idcs = torch.all(data_dimred == nn_point, dim=1).nonzero().item()

                    rand_dimred = data_dimred[rand_idcs]
                    nn_dimred = data_dimred[nn_idcs]

                    quadrant = get_quadrant(rand_dimred, sc) - 1
                    n_instances = quadrant_count[quadrant]
                    if n_instances == 2:
                        continue
                    coords.append([rand_dimred, nn_dimred])
                    cluster_order.append(cluster_i)
                    quadrant_count[quadrant] += 1
                    quadrant_order.append(quadrant)

                    rand_sample = samples[rand_idcs]
                    nn_sample = samples[nn_idcs]
                    image_pair = torch.stack([rand_sample, nn_sample])
                    image_pairs.append(image_pair)
                    clusters_left.remove(cluster_i)

                neighbor_frames = generate(len(image_pairs) * 2, timesteps, torch.cat(image_pairs, dim=0))
                quadrant_tracker = [False, False, False, False]
                quadrant_ax_idcs = []
                for j, q in enumerate(quadrant_order):
                    rand_frame = neighbor_frames[j * 2]
                    nn_frame = neighbor_frames[j * 2 + 1]
                    rand_frame_show = reverse_z_center(rand_frame).permute(1, 2, 0).cpu()
                    nn_frame_show = reverse_z_center(nn_frame).permute(1, 2, 0).cpu()

                    cluster = cluster_order[j]
                    color = scatter_colors[cluster % len(scatter_colors)]

                    if not quadrant_tracker[q]:
                        quadrants[q][0].imshow(rand_frame_show)
                        quadrants[q][1].imshow(nn_frame_show)
                        set_img_border(quadrants[q][0], color, lw)
                        set_img_border(quadrants[q][1], color, lw)

                        quadrant_ax_idcs.append(0)
                        quadrant_tracker[q] = True
                    else:
                        quadrants[q][2].imshow(rand_frame_show)
                        quadrants[q][3].imshow(nn_frame_show)
                        set_img_border(quadrants[q][2], color, lw)
                        set_img_border(quadrants[q][3], color, lw)

                        quadrant_ax_idcs.append(2)

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

                for idx in range(len(coords)):
                    for j in range(2):
                        sc_x_orig, sc_y_orig = coords[idx][j]
                        quadrant_ax_i = quadrant_ax_idcs[idx]
                        bbox = quadrants[quadrant_order[idx]][quadrant_ax_i + j].get_position()

                        position = quadrants[quadrant_order[idx]][quadrant_ax_i + j].get_tightbbox(renderer=None).get_points()
                        window_bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                        w = window_bbox.width * fig.dpi
                        h = window_bbox.height * fig.dpi
                        xmin = window_bbox.xmin * 100
                        ymin = window_bbox.ymin * (100 * (window_bbox.height / window_bbox.width))
                        position = [[(p[0] - xmin) / w, (p[1] - ymin) / h] for p in position]

                        x0, y0 = position[0]
                        x1, y1 = position[1]
                        img_x = (x0 + x1) / 2
                        img_y = (y0 + y1) / 2
                        trans_inset_to_main = sc.transData + ax.transData.inverted()
                        sc_x, sc_y = trans_inset_to_main.transform((sc_x_orig, sc_y_orig))
                        cluster = cluster_order[idx]
                        ax.plot((img_x, sc_x), (img_y, sc_y), color=scatter_colors[cluster % len(scatter_colors)], linewidth=3)
                plt.tight_layout()
                plt.show()
            elif cluster_mode == 'imgs':

                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                x_lim = ax.get_xlim()
                y_lim = ax.get_ylim()
                remove_spines(ax)
                remove_axes(ax)
                idcs_to_show = torch.randint(len(features), (n_imgs_show,))
                samples_to_show = samples[idcs_to_show]
                data_dimred_to_show_all = data_dimred[idcs_to_show]
                img_radius = 0.02

                ax.scatter(data_dimred[:, 0], data_dimred[:, 1], s=0.03, alpha=1.0)

                frames = generate(n_imgs_show, timesteps, samples_to_show)

                for i in range(n_imgs_show):
                    data_dimred_to_show = data_dimred_to_show_all[i]
                    x = data_dimred_to_show[0]
                    y = data_dimred_to_show[1]
                    x_pos = (x - x_lim[0]) / (x_lim[1] - x_lim[0])
                    y_pos = (y - y_lim[0]) / (y_lim[1] - y_lim[0])
                    sc = ax.inset_axes([x_pos - img_radius, y_pos - img_radius, img_radius * 2, img_radius * 2])
                    remove_spines(sc)
                    sc.imshow(reverse_z_center(frames[i]).permute(1, 2, 0).cpu())
                    remove_axes(sc)
                plt.tight_layout()
                plt.show()


            elif cluster_mode == 'steer':
                assert dimred_mode == 'pca', 'Steering only works with PCA'
                direction = dimred.components_[0]
                direction /= torch.norm(direction)

                start_idx = torch.randint(len(features), size=())
                start_feature = features[start_idx].unsqueeze(0)
                start_t = (torch.ones(1, 1).to(device) * t).long().squeeze()
                start_r = samples[start_idx].unsqueeze(0)

                neighbor_frames = generate_from_features(start_feature, t, r, h)
