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

def make_gif(frames, name):
    frame_one = frames[0]
    frame_one.save(f'gifs/{name}.gif', format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

if __name__ == '__main__':
    device = 'cuda:0'
    ckpt_path = './logs/tiles_orig_unet_size_48_steps_1000_lr_0.001/lightning_logs/version_2/checkpoints/last.ckpt'
    model = GenModel.load_from_checkpoint(ckpt_path).to(device)
    model.eval()

    timelapse = True

    timesteps = torch.arange(0, 1000, 10).unsqueeze(1).to(device)

    if timelapse:
        x_pred = model.ddpm.sample_with_timelapse(timestep_interval=10, n=64, size=48)
        x_norm = [reverse_z_center(x) for x in x_pred]

        x_repeat = x_norm[-1].unsqueeze(0).repeat_interleave(50, dim=0)
        x_gif = torch.cat([x.unsqueeze(0) for x in x_norm] + [x_repeat], dim=0)
        x_split = x_gif.split(1, dim=0)
        x_grids = []
        for i in range(len(x_split)):
            x_grid = vutils.make_grid(x_split[i][0], nrow=8)
            x_grids.append(x_grid)

        make_gif([T.ToPILImage()(x.detach().cpu()) for x in x_grids], name='grid2')
    else:
        generated = model.ddpm.sample(n=64, size=48, c=3)
        grid_img = show_images(generated, rows=8, cols=8, scale=True, show=False)
        grid_img[0].save('preds/pred_grid.png')
