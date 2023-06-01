import copy

import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from utils.display import *
from models import ddpm, tiny_unet, med_unet, orig_unet
from datasets.tiles import TilesDataset
from utils.util import z_center, reverse_z_center
import numpy as np


@torch.no_grad()
def sample(ddpm, sample_size, channel, size):
    frames = []
    ddpm.eval()

    timesteps = list(range(ddpm.timesteps))[::-1]
    sample = torch.randn(sample_size, channel, size, size).to(device)

    for i, t in enumerate(tqdm(timesteps)):
        time_tensor = (torch.ones(sample_size, 1) * t).long().to(device)
        residual = ddpm.reverse(sample, time_tensor)
        sample = ddpm.step(residual, time_tensor[0], sample)

    for i in range(sample_size):
        frames.append(sample[i].detach().cpu())
    return frames


def training_loop(model, dataloader, optimizer, num_epochs, num_timesteps, device='cuda:0'):
    global_step = 0
    losses = []

    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(dataloader):
            if isinstance(batch, tuple):
                batch = batch[0]
            batch = batch.to(device)
            noise = torch.randn_like(batch).to(device)
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
size = 48
c = 1 if data == 'mnist' else 3
timesteps = 1000
learning_rate = 1e-3
num_epochs = 500
make_gif = False

transform = T.Compose([
    T.Resize(size),
    T.ToTensor(),
    T.Lambda(z_center)
])

reverse_transform = T.Compose([
    T.Lambda(reverse_z_center),
    T.Lambda(lambda t: t.permute(1, 2, 0)),
    T.Lambda(lambda t: t * 255.),
    T.Lambda(lambda t: t.detach().cpu().numpy().astype(np.uint8))
])

if data == 'mnist':
    root_dir = './data/mnist'
    dataset = torchvision.datasets.MNIST(root=root_dir, train=True, transform=transform, download=True)
    network = tiny_unet.MyTinyUNet(in_channels=c, n_steps=timesteps, size=size)
elif data == 'cifar10':
    root_dir = './data/cifar10'
    dataset = torchvision.datasets.CIFAR10(root=root_dir, train=True, transform=transform, download=True)
    network = med_unet.UNet(in_channels=c, n_steps=timesteps, size=size)
elif data == 'tiles':
    root_dir = './data/Iznik_tiles'
    dataset = TilesDataset(root_dir, transform=transform)
    network = med_unet.UNet(in_channels=c, n_steps=timesteps, size=size)
    # network = orig_unet.UNet(size, channels=c, dim_mults=(1, 2, 4,))
else:
    raise ValueError(f'Invalid data type: {data}')

dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4096, shuffle=True, num_workers=16)

network = network.to(device)
model = ddpm.DDPM(network, timesteps, beta_start=0.0001, beta_end=0.02, device=device)
model.train()
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

losses = training_loop(model, dataloader, optimizer, num_epochs, timesteps, device=device)
plt.plot(losses)
generated = sample(model, 100, channel=c, size=size)
show_images(generated, 'Final result', scale=True)

if make_gif:
    import matplotlib.animation as animation

    rand_idx = 7

    fig = plt.figure()
    ims = []
    for i in range(timesteps):
        im = plt.imshow(samples[i][rand_idx].reshape(size, size, c), animated=True)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    animate.save('diffusion.gif')
    plt.show()

if data == 'cifar10':
    def make_dataloader(dataset, class_name):
        s_indices = []
        s_idx = dataset.class_to_idx[class_name]
        for i in range(len(dataset)):
            current_class = dataset[i][1]
            if current_class == s_idx:
                s_indices.append(i)
        s_dataset = Subset(dataset, s_indices)
        return torch.utils.data.DataLoader(dataset=s_dataset, batch_size=512, shuffle=True)


    class_name = 'horse'
    class_loader = make_dataloader(dataset, class_name)
    tuned_network = copy.deepcopy(network)
    tuned_model = ddpm.DDPM(tuned_network, timesteps, beta_start=0.0001, beta_end=0.02, device=device)
    num_epochs = 100
    timesteps = model.timesteps
    learning_rate = 1e-3
    tuned_model.train()
    optimizer = torch.optim.Adam(tuned_model.parameters(), lr=learning_rate)
    training_loop(tuned_model, class_loader, optimizer, num_epochs, timesteps, device=device)
    generated = sample(tuned_model, 100, c, size)
    show_images(generated, f'Generated {class_name}', True)
