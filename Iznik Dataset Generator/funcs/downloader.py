from bing_image_downloader import downloader
import os
from PIL import Image
import hashlib
import random

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
   
    def calculate_hash(self, file_path):
        with open(file_path, 'rb') as f:
            image_data = f.read()
            hash_value = hashlib.md5(image_data).hexdigest()
        return hash_value

    def delete_duplicatas(self):
        image_dict = {}
        for file_name in os.listdir(self.download_path):
            file_path = os.path.join(self.download_path, file_name)
            if os.path.isfile(file_path):
                hash_value = self.calculate_hash(file_path)
                if hash_value in image_dict:
                    # Duplicate image found
                    os.remove(file_path)
                    #print(f"Duplicate image deleted: {file_path}")
                else:
                    # Unique image
                    image_dict[hash_value] = file_path

    def save_random_flip(self, image_path):
        # Open the image
        image = Image.open(image_path)
        
        # Randomly select the flip types
        flip_types = random.choices(["horizontal", "vertical", "both"])
        
        # Perform the flips
        flipped_image = image
        for flip_type in flip_types:
            if flip_type == "horizontal":
                flipped_image = flipped_image.transpose(Image.FLIP_LEFT_RIGHT)
            elif flip_type == "vertical":
                flipped_image = flipped_image.transpose(Image.FLIP_TOP_BOTTOM)
            elif flip_type == "both":
                flipped_image = flipped_image.transpose(Image.FLIP_TOP_BOTTOM)
                flipped_image = flipped_image.transpose(Image.FLIP_LEFT_RIGHT)
                
        # Get the original image name
        image_name = os.path.basename(image_path)
        
        # Construct the flipped image path
        flipped_image_path = os.path.join(self.download_path, "flipped_" + "_".join(flip_types) + "_" + image_name)
        
        # Save the flipped image
        flipped_image.save(flipped_image_path)
        
        #print("Flipped image saved:", flipped_image_path)
        

    def random_dataset_modif(self):
        if self.flips:
            for file_name in os.listdir(self.download_path):
                file_path = os.path.join(self.download_path, file_name)
                is_flipped = random.randint(0,1)
                
                if is_flipped:
                    self.save_random_flip(file_path)               
                    
    def download_dataset(self):
        
        self.ask_user()
        self.download()
        self.treat_images()
        self.delete_duplicatas()
        self.random_dataset_modif()
        self.delete_duplicatas()
        self.sort_images()