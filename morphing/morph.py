import argparse
import os
import pickle
import re
import imageio
import sys
sys.path.append("..")
import numpy as np
from PIL import Image
import math
import main_tools.loader as loader
import dnnlib
import torch
import tensorflow as tf

device = torch.device('cuda')
#----------------------------------------------------------------------------

def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros(list(images.shape[1:-2]) + [grid_h * img_h, grid_w * img_w], dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[..., y : y + img_h, x : x + img_w] = images[idx]
    return grid

def convert_images_to_uint8(images, drange=[-1,1], nchw_to_nhwc=False, shrink=1):
    """Convert a minibatch of images from float32 to uint8 with configurable dynamic range.
    Can be used as an output transformation for Network.run().
    """
    images = tf.cast(images, tf.float32)
    if shrink > 1:
        ksize = [1, 1, shrink, shrink]
        images = tf.nn.avg_pool(images, ksize=ksize, strides=ksize, padding="VALID", data_format="NCHW")
    if nchw_to_nhwc:
        images = tf.transpose(images, [0, 2, 3, 1])
    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)
    return tf.saturate_cast(images, tf.uint8)


def generate_images(network_pkl, truncation_psi, outdir, class_idx, dlatents1_npz, dlatents2_npz, name="morph"):

    print("Loading network...")
    Gs = loader.load_network(network_pkl, eval=True)["Gs"].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)
    start = np.load(dlatents1_npz)
    end = np.load(dlatents2_npz)
    transition_frames = 300
    mp4_fps = 60
    grid_size = [1,1]
    image_zoom = 1

    writer = imageio.get_writer(f'{outdir}/{name}.mp4', mode='I', fps=60, codec='libx264', bitrate='16M')

    for frame_idx in range(transition_frames):
        #frame_idx = int(np.clip(np.round(t * mp4_fps), 0, transition_frames - 1))

        section = frame_idx // transition_frames

        transition_i = frame_idx - (transition_frames * section)
        maxindex = transition_frames-1.0
        # mu1 = min(max(0, (transition_i*1.0/maxindex) ), 1)                             # linear interpolation
        # mu1 = min(max(0, (transition_i*1.0/maxindex)*(transition_i*1.0/maxindex) ), 1) # quadratic interpolation
        mu1 = min(max(0, 1 - math.cos(math.pi * transition_i / maxindex)), 2) / 2  # sine interpolation
        lat = np.multiply(start, 1.0-mu1)+ np.multiply(end, mu1)
        labels = np.zeros([lat.shape[0], 0], np.float32)
        images = Gs.synthesis(lat, output_transform=dict(func=convert_images_to_uint8, nchw_to_nhwc=True), randomize_noise=False)
        writer.append_data(images[0])


#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser.add_argument('--dlatents1', dest='dlatents1_npz')
    parser.add_argument('--dlatents2', dest='dlatents2_npz')
    parser.add_argument('--name', dest='name')
    #g.add_argument('--dlatents3', dest='dlatents_npz', help='Generate images for saved dlatents')
    #g.add_argument('--dlatents4', dest='dlatents_npz', help='Generate images for saved dlatents')
    #g.add_argument('--dlatents5', dest='dlatents_npz', help='Generate images for saved dlatents')
    parser.add_argument('--trunc', dest='truncation_psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser.add_argument('--class', dest='class_idx', type=int, help='Class label (default: unconditional)')
    parser.add_argument('--outdir', help='Where to save the output images', required=True, metavar='DIR')

    args = parser.parse_args()
    generate_images(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------