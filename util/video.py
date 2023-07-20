import imageio.v2 as imageio
import os
import argparse
from PIL import Image
import imageio
import numpy as np
import os
import sys
import subprocess


#Choose with parser which type of video you want to make
parser = argparse.ArgumentParser(description='All TileGAN functionalities')
parser.add_argument("--gif", "-g", help="Generate gif", default=None, action="store_true")
parser.add_argument("--mp4", "-m", help="Generate mp4", default=None, action="store_true")
args = parser.parse_args()

def mp4making(image_folder, result_name):

    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # Sort image files in ascending order
    image_files = sorted(image_files)

    # Read images into array
    images = []
    for file in image_files:
        images.append(imageio.imread(file))

    #Add a progress bar while the gif is created
    
    # Save as GIF 
    imageio.mimsave(os.path.join('results/videos',result_name), images, fps=10)
    
def gifmaking(result_name):
    
    image_folder = 'gan/results/exp-000/visuals/images'
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    image_files = sorted(image_files)
    images = [Image.open(file) for file in image_files]

    width, height = 320, 213
    images = [image.resize((width, height), Image.LANCZOS) for image in images]
    
    imageio.mimwrite(os.path.join('results/videos', result_name), images, format='GIF', duration=0.1) #type: ignore
    
if __name__ == "__main__":
    if args.gif:
        gifmaking(result_name="image.gif")
    if args.mp4:
        mp4making("results/images", result_name="image.mp4")