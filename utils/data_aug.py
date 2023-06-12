import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class Augmentation:
    def __init__(self, mode, is_rotating):
        self.mode = mode
        if mode == 'flip':
            self.fn = Flip(include_diagonals=not is_rotating)
        elif mode == 'rotate':
            self.fn = Rotate()
        elif mode == 'symmetry':
            self.fn = Symmetry()
        else:
            raise ValueError(f'{mode} not a valid augmentation mode')

    def __call__(self, sample):
        return self.fn(sample)


class Chain(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, sample):
        sample = sample.unsqueeze(0)
        sample_orig = sample.clone()
        for augmentation in self.augmentations:
            sample = augmentation(sample)
        sample = torch.cat([sample_orig, sample], dim=0)
        return sample


class Flip(object):
    def __init__(self, include_diagonals=True):
        super().__init__()
        self.include_diagonals = include_diagonals

    def __call__(self, sample):
        x_axis_flip = sample.flip(2)
        y_axis_flip = sample.flip(3)
        samples = [x_axis_flip, y_axis_flip]
        if self.include_diagonals:
            # these augmentations are redundant if also rotating
            pos_axis_flip = y_axis_flip.rot90(dims=(2, 3))
            neg_axis_flip = x_axis_flip.rot90(dims=(2, 3))
            samples.extend([pos_axis_flip, neg_axis_flip])
        return torch.cat(samples, dim=0).unique(dim=0)


class Rotate(object):
    def __init__(self):
        super().__init__()
        self.angles = [0, 90, 180, 270]

    def __call__(self, sample):
        samples = []
        for angle in self.angles:
            rotated_x = TF.rotate(sample, angle, expand=True)
            if angle % 90 != 0:
                orig_s = rotated_x.shape[-1]
                new_s = int((orig_s / 2) // 2)
                rotated_x = rotated_x[:, new_s:(orig_s - new_s), new_s:(orig_s - new_s)]
            samples.append(rotated_x)
        return torch.cat(samples, dim=0).unique(dim=0)


class Symmetry(object):
    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        s = sample.shape[-1]

        x_axis_top_symmetry = sample.clone()
        x_axis_top_symmetry[:, :, s // 2:, :] = sample[:, :, :s // 2, :].flip(2)

        x_axis_bottom_symmetry = sample.clone()
        x_axis_bottom_symmetry[:, :, :s // 2, :] = sample[:, :, s // 2:, :].flip(2)

        y_axis_left_symmetry = sample.clone()
        y_axis_left_symmetry[:, :, :, s // 2:] = sample[:, :, :, :s // 2].flip(3)

        y_axis_right_symmetry = sample.clone()
        y_axis_right_symmetry[:, :, :, :s // 2] = sample[:, :, :, s // 2:].flip(3)

        tri_lower = sample.rot90(dims=(2, 3)).tril(diagonal=-1)
        tri_upper = TF.rotate(sample, 180).triu(diagonal=0)
        pos_axis_triu_symmetry = TF.rotate((tri_lower + tri_upper), 270)

        tri_upper = sample.rot90(dims=(2, 3)).triu(diagonal=0)
        tri_lower = TF.rotate(sample, 180).tril(diagonal=-1)
        pos_axis_tril_symmetry = TF.rotate((tri_lower + tri_upper), 270)

        tri_lower = sample.tril(diagonal=-1)
        tri_upper = TF.rotate(sample, 180).triu(diagonal=0)
        neg_axis_tril_symmetry = tri_lower + tri_upper

        tri_upper = sample.triu(diagonal=0)
        tri_lower = TF.rotate(sample, 180).tril(diagonal=-1)
        neg_axis_triu_symmetry = tri_lower + tri_upper

        samples = [x_axis_top_symmetry, x_axis_bottom_symmetry,
                   y_axis_left_symmetry, y_axis_right_symmetry,
                   pos_axis_tril_symmetry, pos_axis_triu_symmetry,
                   neg_axis_tril_symmetry, neg_axis_triu_symmetry]

        return torch.cat(samples, dim=0).unique(dim=0)
