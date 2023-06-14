import matplotlib.pyplot as plt
from utils.util import reverse_z_center
import numpy as np
import torch

def show_images(images, rows, cols, title="", scale=False, show=True):
    """Shows the provided images as sub-pictures in a square"""
    if scale:
        images = [reverse_z_center((im.permute(1, 2, 0)).numpy()) for im in images]

    f, ax = plt.subplots(rows, cols, figsize=(cols, rows))

    for r in range(rows):
        for c in range(cols):
            ax[r][c].imshow(images[r * cols + c])

    remove_axes(ax)
    # f.subplots_adjust(hspace=-0.1, wspace=-0.1)
    f.tight_layout()
    if show:
        plt.show()
    else:
        return f, ax

# source: https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
def show_noise_steps(imgs, orig, row_title=None, scale=False, show=True, **imshow_kwargs):
    # if not isinstance(imgs[0], list):
    #     # Make a 2d grid even if there's just 1 row
    #     imgs = [imgs]
    if scale:
        imgs = [reverse_z_center(im) for im in imgs]
        orig = reverse_z_center(orig)

    num_rows = len(imgs)
    with_orig = orig is not None
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(num_cols * 5, num_rows * 5), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = list(torch.split(row, 1, dim=0))
        row = [orig[row_idx]] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img.squeeze().permute(1, 2, 0)), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig, axs

def _remove_axes(ax):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])


def remove_axes(axes):
    if isinstance(axes, plt.Axes):
        _remove_axes(axes)
    elif len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)

def remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)