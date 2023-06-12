import torch.nn as nn
import torch
from utils.schedulers import *
from tqdm import tqdm

class DDPM(nn.Module):
    def __init__(self, network, timesteps, beta_start=0.0001, beta_end=0.02, device='cuda:0') -> None:
        super(DDPM, self).__init__()
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps, beta_start, beta_end).to(device)
        self.alphas = 1.0 - self.betas
        if device == 'mps':
            self.alphas = self.alphas.cpu()
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(device)
        self.alphas = self.alphas.to(device)
        self.network = network
        self.device = device
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5  # used in add_noise
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5  # used in add_noise and step

    def add_noise(self, x_start, x_noise, timesteps):
        # The forward process
        # x_start and x_noise (bs, n_c, w, d)
        # timesteps (bs)
        s1 = self.sqrt_alphas_cumprod[timesteps]  # bs
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]  # bs
        s1 = s1.reshape(-1, 1, 1, 1)  # (bs, 1, 1, 1) for broadcasting
        s2 = s2.reshape(-1, 1, 1, 1)  # (bs, 1, 1, 1)
        return s1 * x_start + s2 * x_noise

    def reverse(self, x, t):
        # The network return the estimation of the noise we added
        return self.network(x, t)

    @torch.no_grad()
    def sample(self, n, size, c=3):
        frames = []
        self.eval()

        timesteps = list(range(self.timesteps))[::-1]
        sample = torch.randn(n, c, size, size).to(self.device)

        for i, t in enumerate(tqdm(timesteps)):
            time_tensor = (torch.ones(n, 1) * t).long().to(self.device)
            residual = self.reverse(sample, time_tensor)
            sample = self.step(residual, time_tensor[0], sample)

        for i in range(n):
            frames.append(sample[i].detach().cpu())
        return frames

    @torch.no_grad()
    def sample_with_timelapse(self, timestep_interval, n, size, c=3):
        self.eval()

        frames = []
        timesteps = list(range(self.timesteps))[::-1]
        sample = torch.randn(n, c, size, size).to(self.device)

        for i, t in enumerate(tqdm(timesteps)):
            time_tensor = (torch.ones(n, 1) * t).long().to(self.device)
            residual = self.reverse(sample, time_tensor)
            sample = self.step(residual, time_tensor[0], sample)
            if t % timestep_interval == 0 or t == self.timesteps - 1:
                frames.append(sample.detach().cpu())
        return frames


    def step(self, model_output, timestep, sample):
        # one step of sampling
        # timestep (1)
        t = timestep
        coef_epsilon = (1 - self.alphas) / self.sqrt_one_minus_alphas_cumprod
        coef_eps_t = coef_epsilon[t].reshape(-1, 1, 1, 1)
        coef_first = 1 / self.alphas ** 0.5
        coef_first_t = coef_first[t].reshape(-1, 1, 1, 1)
        pred_prev_sample = coef_first_t * (sample - coef_eps_t * model_output)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output).to(self.device)
            variance = ((self.betas[t] ** 0.5) * noise)

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample