import torch

def z_center(x):
    return (x * 2) - 1

def reverse_z_center(x):
    return (x + 1) / 2