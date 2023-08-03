import os
from PIL import Image
import shutil

class Results:
    def __init__(self, results_path):
        self.results_path = results_path
        os.makedirs(self.results_path, exist_ok=True)

    def get_images(self, folder_path):
        # Read the image files from the folder and return a list of PIL.Image objects
        images = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.jpg') or file_name.endswith('.png') or file_name.endswith('.jpeg'):
                file_path = os.path.join(folder_path, file_name)
                image = Image.open(file_path)
                images.append(image)
        return images

    def save_grid(self, images, grid_size=(4, 4), file_name='grid.jpg'):
        # Create a grid of images and save it to a file
        grid = Image.new('RGB', (images[0].width * grid_size[1], images[0].height * grid_size[0]))
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                grid.paste(images[i * grid_size[1] + j], (j * images[0].width, i * images[0].height))
        grid.save(os.path.join(self.results_path, file_name), 'JPEG', quality=80)
        
    def save_image(self, image, file_name):
        # Save the image to a file
        image.save(os.path.join(self.results_path, file_name), 'JPEG', quality=80)
        
    def copy_file(self, file_path, folder_destination):
        # Copy a file to the results folder
        file_name = os.path.basename(file_path)
        shutil.copy(file_path, os.path.join(folder_destination, file_name))
    
    def copy_files(self, folder_destination):
            for file_name in os.listdir(self.results_path):
                if (file_name.endswith('.jpg') or file_name.endswith('.png') or file_name.endswith('.jpeg')):
                    self.copy_file(os.path.join(self.results_path,file_name), folder_destination)

def add_index_to_filename(filename):
    base, ext = os.path.splitext(filename)
    index = 0
    new_filename = filename

    while os.path.exists(new_filename):
        index += 1
        new_filename = f"{base}_{index}{ext}"

    return new_filename

#For every file in the folder, change its end filename to .png
def convert_to_png(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
            base, ext = os.path.splitext(file_name)
            new_file_name = base + '.png'
            os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))
            