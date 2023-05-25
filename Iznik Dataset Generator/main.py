from funcs.downloader import Downloader
from funcs.dataset import IznikDataset
import funcs.interface as interface
import funcs.move_dataset as md

def main():
    
    # Create downloader class
    downloader = Downloader()
    
    # Get the images
    downloader.download_dataset()
    
    # Create the dataset, which is empty as first
    dataset = IznikDataset()

    while len(dataset) != len(dataset.images_list):
        
        # Get the top 25 images
        E = dataset.top25()
        dataset.images_studied +=E
        #print(f'E = {E}\n')
        
        # Display them and select their labels
        Y = interface.display_and_label(E, downloader.download_path)
        dataset.labels_studied +=Y
        #print(f'Y = {Y}\n')
        
        if len(dataset) != len(dataset.images_list):
            dataset.learn()    
    
    md.move_good_images()         
    print("*** DATASET COMPLETED ***")
    
    
    
if __name__ == '__main__':
    main()