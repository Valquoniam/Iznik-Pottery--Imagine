import torch
from pytorch_lightning import seed_everything
import logging

def z_center(x):
    return (x * 2) - 1

def reverse_z_center(x):
    return (x + 1) / 2

def scale(x):
    x_min = x - x.min()
    return x_min / x_min.max()

def seed_worker(worker_id):
    clear_logging()
    worker_seed = torch.initial_seed() % 2 ** 32
    seed_everything(worker_seed)

def clear_logging():
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict if 'lightning' in name]
    for logger in loggers:
        logger.setLevel(logging.ERROR)