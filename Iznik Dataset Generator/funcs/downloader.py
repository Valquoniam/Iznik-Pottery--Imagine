from bing_image_downloader import downloader
import os
from PIL import Image
import random
import numpy as np


class Downloader:
    def __init__(self):
        self.search_query = 'Iznik_pottery_tiles'
        self.download_path = self.search_query
        self.number_of_items = 0
        self.flips = 0

    def ask_user(self):
        # Ask the user if he wants to download the dataset from google
        user_input = input("\nDo you want to download a set of images? (Y/n) \n")
    
        # If yes, note the informations
        if user_input.lower() == "y" or user_input.lower() == "yes":
            self.number_of_items = int(input("\nHow many images do you want to download ? \n"))  
            self.flips = 1
        elif user_input.lower() == "n" or user_input.lower() == "no":
            self.flips = 0
        else:
            exit('Wrong answer')    
    
    def download(self):
        if self.number_of_items >0:
            downloader.download(self.search_query,output_dir='../Iznik Dataset Generator',  limit=self.number_of_items, adult_filter_off=True, force_replace=True, timeout=5, verbose=False)
            
    def treat_images(self):
        for filename in os.listdir(self.download_path):
            
            img = Image.open(os.path.join(self.download_path, filename))
            img = img.convert('RGB')
            new_resolution = (256, 256)
            img_resized = img.resize(new_resolution, resample=Image.Resampling.BOX)
            img_resized.save(os.path.join(self.download_path, filename))
    
    def sort_images(self):
        for i,file_name in enumerate(os.listdir(self.download_path)): 
            if not((file_name.endswith(".jpeg") or file_name.endswith(".jpg") or file_name.endswith(".png"))):
                    os.remove(os.path.join(self.download_path, file_name)) 
        
        for i,file_name in enumerate(os.listdir(self.download_path)):            
                    new_file_name = f"img_lolilol{i:05d}.jpeg"
                    new_path = os.path.join(self.download_path, new_file_name)
                    os.rename(os.path.join(self.download_path, file_name),new_path)
                    
        for i,file_name in enumerate(os.listdir(self.download_path)):            
                    new_file_name = f"img_{i:05d}.jpeg"
                    new_path = os.path.join(self.download_path, new_file_name)
                    os.rename(os.path.join(self.download_path, file_name),new_path)


    def delete_duplicatas(self):
        hash_values = {}
        for filename in os.listdir(self.download_path):
            
            # Chemin complet de l'image
            image_path = os.path.join(self.download_path, filename)
            image = Image.open(image_path)
            
            # Redimensionner l'image en 16x16 pixels
            image = image.resize((50, 50))
            image_array = np.array(image)
            flattened_array = image_array.flatten()
            binary_array = np.packbits(flattened_array)
            hash_value = binary_array.tobytes().hex()

            # Vérifier si le hash value est déjà dans le dictionnaire
            if hash_value in hash_values:
                os.remove(image_path)
                print("Image en doublon supprimée:", filename)
            else:
                hash_values[hash_value] = filename

                print("Image traitée:", filename)        
                    
    def download_dataset(self):
        
        self.ask_user()
        self.download()
        self.treat_images()
        self.delete_duplicatas()
        self.delete_duplicatas()
        self.sort_images()