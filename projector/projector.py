import sys
sys.path.append("../")
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Normalize
import numpy as np
import main_tools.loader as loader
import imageio
from tqdm import tqdm
from training import misc
from training.misc import crop_max_rectangle as crop
from util.results import add_index_to_filename
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

# Utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0 #type: ignore
model = "../training_results/iznik_snapshot.pkl"
target_image = "../results/images/img_0027.png"

# Load the target image and preprocess it
def load_and_preprocess_image(image_path, image_size):
    # Chargement de l'image depuis le fichier
    img = Image.open(image_path)

    # Redimensionnement de l'image à la taille requise pour le GANformer
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # Si votre modèle nécessite une normalisation spécifique, ajoutez-la ici
    ])

    # Appliquer les transformations à l'image et la convertir en tenseur
    image_tensor = transform(img).unsqueeze(0).to(device)  # type: ignore # Move to the selected device
    return image_tensor

# Load generator
print("Loading network...")
G = loader.load_network(model, eval=True)["Gs"].to(device)  # type: ignore

# Load pre-trained DINO v2 model
dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)  # type: ignore

# Load LPIPS model
#lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device)

# Get parameters from DINO layer norm
dino_mean = dino.norm.weight.data.mean()
dino_std = dino.norm.weight.data.std()
target_img = load_and_preprocess_image(target_image, (256, 256))

# Normalization for DINO
dino_normalization = Normalize(mean=dino_mean, std=dino_std)

# Define GAN inversion model
class InvModel(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, x):
        feats = self.feature_extractor(dino_normalization(x))
        return feats

inv_model = InvModel(dino)

# Initial random noise vector
latent = torch.randn([1, *G.input_shape[1:]], device=device, requires_grad=True)

# Parameters for noise addition
noise_std = 0.01

# Optimize to match target image features
target_feats = inv_model(target_img).to(device)
optimizer = torch.optim.Adam([latent], lr=0.01) # type: ignore
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.1, verbose=True) # type: ignore

generated_images = []
best_loss = float('inf')
converged = False

for i in range(1000):
    optimizer.zero_grad()
    
     # Add noise to the latent vector
    noisy_latent = latent + torch.randn_like(latent) * noise_std
    
    generated_image = G(noisy_latent)[0]
    generated_image_rendered= generated_image.detach().cpu().numpy()
    pil_image= crop(misc.to_pil(generated_image_rendered[0]), 1.0)
    generated_images.append(pil_image)
    feats = inv_model(generated_image).to(device)
    l1_loss = nn.L1Loss()(feats, target_feats)
    loss = l1_loss

    if loss < best_loss:
        best_loss = loss
        best_image = generated_image.clone()

    loss.backward(retain_graph=True)
    optimizer.step()


    # Print loss advancement
    if i % 100 == 0:
        print('Step {} : loss {}'.format(i, loss.item()))

    """# Early stopping
    if i > 200 and lpips_loss.item() >= 0.95 * best_loss.item():
        converged = True
        break"""

# Use the best image if converged
if converged:
    generated_image = best_image #type: ignore

with imageio.get_writer(add_index_to_filename("../results/videos/projection.mp4"), mode='I', fps=60) as writer:
    for generated_image in tqdm(generated_images):  # type: ignore
        pil_image = generated_image.convert("RGB")
        writer.append_data(np.array(pil_image))  # type: ignore

# Generated image
generated_image = G(latent, truncation_psi=1.0)[0].detach().cpu().numpy()

# Save image
img = crop(misc.to_pil(generated_image[0]), 1.0)
img.save("gen_via_dino.png")
