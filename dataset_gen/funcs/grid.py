import os
import math
from PIL import Image
import matplotlib.pyplot as plt

def make_grid(num_images):
    # Calculate the nearest square number
    grid_size = math.ceil(math.sqrt(num_images))
    
    # Get the filenames of images in the folder
    image_folder = "../found_images"
    image_filenames = sorted([filename for filename in os.listdir(image_folder) if filename.endswith(".jpeg") or filename.endswith(".png")])
    
    # Create a new blank canvas for the grid
    grid_width = grid_size * max(Image.open(os.path.join(image_folder, filename)).width for filename in image_filenames)
    grid_height = grid_size * max(Image.open(os.path.join(image_folder, filename)).height for filename in image_filenames)
    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    # Paste images onto the grid
    for i in range(num_images):
        row = i // grid_size
        col = i % grid_size
        image_path = os.path.join(image_folder, image_filenames[i])
        img = Image.open(image_path)
        img = img.resize((grid_width // grid_size, grid_height // grid_size), Image.LANCZOS)
        grid.paste(img, (col * img.width, row * img.height))
    
    # Display or save the grid
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

# Call the function with the number of images you have
num_images = 100 # Change this to the number of images you have
make_grid(num_images)