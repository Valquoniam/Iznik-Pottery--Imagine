import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import os
from torch.utils.data import Dataset
from funcs.downloader import Downloader
from contextlib import contextmanager
import sys
from tqdm import tqdm

# Pour éviter le message relou 'No module triton detected'
#######################
@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

with suppress_stdout_stderr():
    pretrained_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda()
#######################
   
downloader = Downloader()
device = "cuda:0"


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class IznikDataset(Dataset):
    
    def __init__(self):
        
        super(IznikDataset, self).__init__()
        self.transform = transforms.Compose([
                         transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        self.linear = nn.Linear(384, 1, bias=False).cuda()
        
        with torch.no_grad( ):
            
            self.images_list = [image for image in os.listdir('Iznik_pottery_tiles')]
            self.images_opened = [Image.open(os.path.join(downloader.download_path,image)) for image in self.images_list]
            self.images_transformed = []

            # Créer une barre de progression avec le nombre total d'itérations
            progress_bar = tqdm(total=len(self.images_opened), desc='Passing images through DINOv2...')  
            
            for image in self.images_opened:
                result = pretrained_model(self.transform(image).unsqueeze(0).to(device)).squeeze()
                self.images_transformed.append(result)
                torch.cuda.empty_cache()
                progress_bar.update(1)
                
            progress_bar.close()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.linear.parameters(), lr=0.01)
            
        self.labels_studied = []
        self.images_studied = []

    def __len__(self):
        return len(self.images_studied)
    
    def __getitem__(self, idx):
        
        dictionary = {'images' : self.images_studied[idx : idx +25] , 'labels' : self.labels_studied[idx : idx + 25]}
        return dictionary
    
    def forward(self, x):
        out = self.linear(x)
        out = nn.Sigmoid()(out)
        return out
      
    def predict(self):
        results = []
        # Predict for non-studied-yet elements
        for i in range(len(self.images_transformed)):
            if self.images_list[i] not in self.images_studied:
                results.append((self.forward(self.images_transformed[i]), i))    

        return results
    
    def top25(self):
        results = self.predict()
        results_1st_elem = torch.tensor([t[0] for t in results])
        sorted_y_hat_indexes = torch.argsort(results_1st_elem, descending = True)
        top25 = [self.images_list[results[i][1]] for i in sorted_y_hat_indexes[:25]]
        return top25
    
    def learn(self):
        
        for i in range(0,len(self.images_studied)-1,24):
            batch = self[i]
            images = torch.stack([self.images_transformed[self.images_list.index(image)] for image in batch['images']]).cuda()
            labels = torch.tensor(batch['labels'], dtype=torch.float32).cuda()
            for epoch in range(100):
                
                # Remise à zéro des gradients
                self.optimizer.zero_grad()

                # Propagation avant (calcul de la prédiction)
                predictions = self.forward(images).squeeze()
                
                # Calcul de la perte
                loss = self.criterion(predictions, labels)

                # Rétropropagation du gradient et mise à jour des poids
                loss.backward(retain_graph=True)
                self.optimizer.step()
                
                #print(f'Loss at epoch {epoch} : {loss.item():.4f}')
                