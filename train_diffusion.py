import copy
import sys
import os
import logging
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
from utils.util import z_center, reverse_z_center, seed_worker
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from utils.data_aug import *


class GenModel(pl.LightningModule):
    def __init__(self, data, model_type, size, in_c, n_steps, lr, device='cuda:0'):
        super().__init__()
        self.data = data
        self.model_type = model_type
        self.size = size
        self.in_c = in_c
        self.n_steps = n_steps
        self.lr = lr
        self.save_hyperparameters()

        if self.model_type == 'tiny_unet':
            self.model = tiny_unet.MyTinyUNet(in_channels=in_c, n_steps=n_steps, size=size)
        elif self.model_type == 'med_unet':
            self.model = med_unet.UNet(in_channels=in_c, n_steps=n_steps, size=size)
        elif self.model_type == 'orig_unet':
            self.model = orig_unet.UNet(size, channels=in_c, dim_mults=(1, 2, 4,))
        else:
            raise ValueError(f'Invalid model type: {self.model_type}')

        self.model.to(device)
        self.ddpm = ddpm.DDPM(self.model, n_steps, beta_start=0.0001, beta_end=0.02, device=device)

    def forward(self, x, noise):
        timesteps = torch.randint(0, self.n_steps, (x.shape[0],)).long().to(noise.device)

        noisy = self.ddpm.add_noise(x, noise, timesteps)
        noise_pred = self.ddpm.reverse(noisy, timesteps)
        return noise_pred

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list):
            batch = batch[0]
        b1, b2, c, h, w = batch.shape
        batch = batch.reshape(-1, c, h, w)
        noise = torch.randn_like(batch).to(batch.device)
        noise_pred = self(batch, noise)
        loss = F.mse_loss(noise_pred, noise)
        self.log('loss/train', loss)

        if self.global_step % 100 == 0 and self.global_step > 0:
            self.eval()
            generated = self.ddpm.sample(64, self.size, c=self.in_c)
            grid_img = show_images(generated, rows=8, cols=8, scale=True, show=False)
            self.logger.experiment.add_figure('pred_grid', grid_img[0], global_step=self.global_step)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return [opt]


if __name__ == '__main__':
    seed = 4321
    seed_everything(seed)
    device = 'cpu'

    data = 'tiles'
    size = 64
    timesteps = 1000
    learning_rate = 1e-3
    epochs = 10000
    log_to_file = False
    batch_size = 1
    augment_modes = ['flip', 'rotate', 'symmetry']

    c = 1 if data == 'mnist' else 3

    augmentations = []
    for mode in augment_modes:
        augmentations.append(Augmentation(mode, is_rotating='rotate' in augment_modes))
    augmentations = Chain(augmentations)

    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        augmentations,
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
        network = 'tiny_unet'
    elif data == 'cifar10':
        root_dir = './data/cifar10'
        dataset = torchvision.datasets.CIFAR10(root=root_dir, train=True, transform=transform, download=True)
        network = 'med_unet'
    elif data == 'tiles':
        root_dir = './data/Iznik_tiles'
        dataset = TilesDataset(root_dir, transform=transform)
        network = 'orig_unet'
    else:
        raise ValueError(f'Invalid data type: {data}')

    g = torch.Generator()
    g.manual_seed(seed)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                         worker_init_fn=seed_worker, generator=g)

    model = GenModel(data=data, model_type=network, size=size, in_c=c, n_steps=timesteps, lr=learning_rate,
                     device=device)

    exp_dir = f'{data}_{network}_size_{size}_steps_{timesteps}_lr_{learning_rate}_aug_{",".join(augment_modes)}'
    logger = TensorBoardLogger(save_dir=os.path.join('./logs/', exp_dir), default_hp_metric=False)

    trainer = Trainer(devices=1,
                      accelerator='gpu' if device == 'cuda' else device,
                      log_every_n_steps=10,
                      logger=logger,
                      max_epochs=epochs,
                      default_root_dir=os.path.join('./logs/', exp_dir),
                      callbacks=ModelCheckpoint(monitor='loss/train',
                                                save_top_k=5,
                                                save_last=True,
                                                filename='{epoch:03d}',
                                                mode='min')
                      )

    eval_root = os.path.join(exp_dir, 'lightning_logs', f'version_{trainer.logger.version}')
    if log_to_file:
        logging.basicConfig(filename=os.path.join(eval_root, 'exp.log'), level=logging.INFO, force=True)
        logging.info('LOGGING')
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.info('PRINTING TO STDOUT')

    logging.info('Arguments: ', exp_dir)

    trainer.fit(model, loader)
