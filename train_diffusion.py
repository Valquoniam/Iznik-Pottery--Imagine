import copy

import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from utils.display import *
from models import ddpm, unet, tiny_unet
from datasets.tiles import TilesDataset

def generate_image(ddpm, sample_size, channel, size):
    """Generate the image from the Gaussian noise"""
    frames = []
    frames_mid = []
    ddpm.eval()
    with torch.no_grad():
        timesteps = list(range(ddpm.num_timesteps))[::-1]
        sample = torch.randn(sample_size, channel, size, size).to(device)

        for i, t in enumerate(tqdm(timesteps)):
            time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)
            residual = ddpm.reverse(sample, time_tensor)
            sample = ddpm.step(residual, time_tensor[0], sample)

            if t == ddpm.num_timesteps // 2:
                for i in range(sample_size):
                    frames_mid.append(sample[i].detach().cpu())

        for i in range(sample_size):
            frames.append(sample[i].detach().cpu())
    return frames, frames_mid

def training_loop(model, dataloader, optimizer, num_epochs, num_timesteps, device='cuda:0'):
    global_step = 0
    losses = []

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(dataloader):
            batch = batch[0].to(device)
            noise = torch.randn(batch.shape).to(device)
            timesteps = torch.randint(0, num_timesteps, (batch.shape[0],)).long().to(device)

            noisy = model.add_noise(batch, noise, timesteps)
            noise_pred = model.reverse(noisy, timesteps)
            loss = F.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(1)
            logs = {"epoch": epoch, "loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1

        progress_bar.close()
    return losses


device = 'cuda'
data = 'tiles'
size = 16

transform01 = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5), (0.5))
])

if data == 'mnist':
    root_dir = './data/mnist'
    dataset = torchvision.datasets.MNIST(root=root_dir, train=True, transform=transform01, download=True)
    unet_choice = tiny_unet.MyTinyUNet
elif data == 'cifar10':
    root_dir = './data/cifar10'
    dataset = torchvision.datasets.CIFAR10(root=root_dir, train=True, transform=transform01, download=True)
    unet_choice = unet.UNet
elif data == 'tiles':
    root_dir = './data/Iznik_tiles'
    dataset = TilesDataset(root_dir, transform=transform01)
    unet_choice = unet.UNet

dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4096, shuffle=True, num_workers=24)

batch = next(iter(dataloader))
if isinstance(batch, list):
    batch = batch[0]
bn = [b for b in batch[:100]]
show_images_rescale(bn, "origin")
learning_rate = 1e-4
num_epochs = 200
num_timesteps = 1000
network = unet_choice(in_channels=bn[0].shape[0], n_steps=num_timesteps, size=size)
network = network.to(device)
model = ddpm.DDPM(network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device)
model.train()
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
losses = training_loop(model, dataloader, optimizer, num_epochs, num_timesteps, device=device)
plt.plot(losses)
generated, generated_mid = generate_image(model, 100, channel=bn[0].shape[0], size=size)
show_images_rescale(generated_mid, "Mid result")
show_images_rescale(generated, "Final result")

print('here')

if data == 'cifar10':
    def make_dataloader(dataset, class_name='horse'):
        s_indices = []
        s_idx = dataset.class_to_idx[class_name]
        for i in range(len(dataset)):
            current_class = dataset[i][1]
            if current_class == s_idx:
                s_indices.append(i)
        s_dataset = Subset(dataset, s_indices)
        return torch.utils.data.DataLoader(dataset=s_dataset, batch_size=512, shuffle=True)

    ship_dataloader = make_dataloader(dataset)
    ship_network = copy.deepcopy(network)
    ship_model = ddpm.DDPM(ship_network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device)
    num_epochs = 100
    num_timesteps = model.num_timesteps
    learning_rate = 1e-3
    ship_model.train()
    optimizer = torch.optim.Adam(ship_model.parameters(), lr=learning_rate)
    training_loop(ship_model, ship_dataloader, optimizer, num_epochs, num_timesteps, device=device)
    generated, generated_mid = generate_image(ship_model, 100, 3, 32)
    show_images_rescale(generated, "Generated horses")
    print('here')