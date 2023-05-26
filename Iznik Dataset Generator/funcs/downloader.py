from bing_image_downloader import downloader
import os
from PIL import Image
import numpy as np
import shutil
import imagehash

class Downloader:
    def __init__(self):
        self.search_query = ['Iznik_pottery_tiles', 'Iznik_Pottery', 'Tuiles_Iznik', 'Echantillon_de_tuile_diznik', 'square_iznik_tiles', 'squared_iznik_mosaic']
        self.download_path = 'All_Downloaded_Images'
        self.number_of_items = 0
        self.flips = 0

    def ask_user(self):
        # Ask the user if he wants to download the dataset from google
        user_input = input("\nDo you want to download a set of images? (Y/n) \n")
    
        # If yes, note the informations
        if user_input.lower() == "y" or user_input.lower() == "yes":
            self.number_of_items = int(input("\nHow many images do you want to download ? \n"))  
            self.flips = 1
            
            if os.path.exists('iznik_labels.csv'):
                os.remove('iznik_labels.csv')
            
        elif user_input.lower() == "n" or user_input.lower() == "no":
            self.flips = 0
        else:
            exit('Wrong answer')    
    
    def download(self):
        if self.number_of_items >0:
            for keyword in self.search_query:
                downloader.download(keyword,output_dir='../Iznik Dataset Generator',  limit=self.number_of_items, adult_filter_off=True, force_replace=True, timeout=5, verbose=False)
            
    def merge_folders(self, source_folder2, destination_folder):

        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            
        for filename in os.listdir(source_folder2):
            source_path = os.path.join(source_folder2, filename)
            destination_path = os.path.join(destination_folder, filename)

            if os.path.exists(destination_path):
                base_name, extension = os.path.splitext(filename)
                counter = 1
                while True:
                    
                    new_filename = f"{base_name}({counter}){extension}"
                    new_destination_path = os.path.join(destination_folder, new_filename)
                    
                    if not os.path.exists(new_destination_path):
                        destination_path = new_destination_path
                        break
                    
                    counter += 1

            shutil.copy(source_path, destination_path)
            
    def merge_everything(self):
        if self.flips:
            for folder in self.search_query:
                self.merge_folders(folder, self.download_path)
                shutil.rmtree(folder)
                
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


    def alpharemover(self, image):
        if image.mode != 'RGBA':
            return image
        canvas = Image.new('RGBA', image.size, (255,255,255,255))
        canvas.paste(image, mask=image)
        return canvas.convert('RGB')

    def with_ztransform_preprocess(self,hashfunc, hash_size):
        def function(path):
            image = self.alpharemover(Image.open(path))
            image = image.convert("L").resize((hash_size, hash_size), Image.ANTIALIAS)
            data = image.getdata()
            quantiles = np.arange(100)
            quantiles_values = np.percentile(data, quantiles)
            zdata = (np.interp(data, quantiles_values, quantiles) / 100 * 255).astype(np.uint8)
            image.putdata(zdata)
            return hashfunc(image)
        return function
        
    def delete_duplicatas(self):
        hash_dict = {}
        for file in os.listdir(self.download_path):
            file_path = os.path.join(self.download_path, file)
            hash = self.with_ztransform_preprocess(imagehash.dhash, hash_size = 3)(file_path)
            
            if hash in hash_dict:
                os.remove(file_path)
                print("Image en doublon supprimée:", file)
            else:
                hash_dict[hash] = file
                
    def download_dataset(self):
        
        self.ask_user()
        self.download()
        self.merge_everything()
        self.treat_images()
        self.sort_images()
        self.delete_duplicatas()
        self.sort_images()