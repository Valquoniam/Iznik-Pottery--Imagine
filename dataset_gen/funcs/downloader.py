from bing_image_downloader import downloader
import os
from PIL import Image
import numpy as np
import shutil
import imagehash

class Downloader:
    def __init__(self):
        self.search_query = ['Iznik_tiles', 'iznik_pottery', 'tuile_ottomane_dessin_mosaique', 'tuile_ottomane_dessin_mosaique_carrée']
        # path to the parent folder of the dataset
        self.download_path = "../"+os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))+"/found_images"
        
        self.number_of_items = 0
        self.flips = 0
        
        if not os.path.exists(self.download_path):
            os.mkdir(self.download_path)
            
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
    
    # To download 
    def download(self):
        if self.number_of_items >0:
            for keyword in self.search_query:
                downloader.download(keyword,output_dir="../"+os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  limit=self.number_of_items, adult_filter_off=True, force_replace=True, timeout=5, verbose=False)
    
    # To merge all the images we obtained in one folder (each request creates a new folder, so we need to merge them all into one)
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

            shutil.move(source_path, destination_path)
    
    # Using precedent function but for all our folders  
    def merge_everything(self):
            for folder in self.search_query:
                if os.path.exists(folder):
                    self.merge_folders(folder, self.download_path)
                    shutil.rmtree(folder)
    
    # Resizing images and convert to RGB
    def treat_images(self,folder):
        for filename in os.listdir(folder):
            print(filename)
            img = Image.open(os.path.join(folder, filename))
            img = img.convert('RGB')
            new_resolution = (256, 256)
            img_resized = img.resize(new_resolution, resample=Image.Resampling.BOX)
            img_resized.save(os.path.join(folder, filename))
    
    # Rename properly every image
    def sort_images(self,folder):
        for i,file_name in enumerate(os.listdir(folder)): 
            if not((file_name.endswith(".jpeg") or file_name.endswith(".jpg") or file_name.endswith(".png"))):
                    os.remove(os.path.join(folder, file_name)) 
        
        for i,file_name in enumerate(os.listdir(folder)):            
                    new_file_name = f"img_lolilol{i:05d}.jpeg"
                    new_path = os.path.join(folder, new_file_name)
                    os.rename(os.path.join(folder, file_name),new_path)
                    
        for i,file_name in enumerate(os.listdir(folder)):            
                    new_file_name = f"img_{i:05d}.jpeg"
                    new_path = os.path.join(folder, new_file_name)
                    os.rename(os.path.join(folder, file_name),new_path)


    # Attempt to remove duplicates
    
    ##############################################
    def alpharemover(self, image):
        if image.mode != 'RGBA':
            return image
        canvas = Image.new('RGBA', image.size, (255,255,255,255))
        canvas.paste(image, mask=image)
        return canvas.convert('RGB')

    def with_ztransform_preprocess(self,hashfunc, hash_size):
        def function(path):
            image = self.alpharemover(Image.open(path))
            image = image.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
            data = image.getdata()
            quantiles = np.arange(100)
            quantiles_values = np.percentile(data, quantiles)
            zdata = (np.interp(data, quantiles_values, quantiles) / 100 * 255).astype(np.uint8)
            image.putdata(zdata)
            return hashfunc(image)
        return function
        
    def delete_duplicatas(self, folder):
        for i in range(10):
            hash_dict = {}
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                hash = self.with_ztransform_preprocess(imagehash.dhash, hash_size = 3)(file_path)
                
                if hash in hash_dict:
                    os.remove(file_path)
                    print("Image en doublon supprimée:", file)
                else:
                    hash_dict[hash] = file
                    
    def calculate_average_hash(self,image_path):
        with Image.open(image_path) as img:
            # Resize the image to a fixed size for consistent hashing
            img = img.resize((32, 32), Image.LANCZOS)
            # Convert the image to grayscale
            img = img.convert("L")
            # Calculate the average hash
            img_hash = imagehash.average_hash(img)
        return img_hash

    def find_image_files(self,directory):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif']  # Add more extensions if needed
        image_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                _, ext = os.path.splitext(file)
                if ext.lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
        
        return image_files

    def del_duplicate_image_groups(self,directory):
        image_files = self.find_image_files(directory)
        image_hashes = {}
        hashes = []
        for _,file_path in enumerate(image_files):
            img_hash = self.calculate_average_hash(file_path)
            hamming_distances = [abs(img_hash - hashs) for hashs in hashes if hashes != []]
            #print(f'Hamming distances : {hamming_distances}')
            hashes.append(img_hash)
            if img_hash in image_hashes or (hamming_distances != [] and min(hamming_distances) < 2):
                # Add the file to the list of similar images
                os.remove(file_path)
                print('deleted duplicata')
            else:
                # Create a new list with the current file as the first similar image
                image_hashes[img_hash] = [file_path]
        
        # Find d
    ##############################################            
    
    # Doing verything we need 
    def download_dataset(self):
        
        self.ask_user()
        self.download()
        self.merge_everything()
        self.treat_images(self.download_path)
        self.sort_images(self.download_path)
        self.delete_duplicatas(self.download_path)
        for i in range(10):
            self.del_duplicate_image_groups(self.download_path)
        self.sort_images(self.download_path)