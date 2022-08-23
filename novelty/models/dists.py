import math

import torch


def benford_dist(digit, base=11):
    return torch.log(1 + 1/(digit+1))/math.log(base)


def instrumental_dist(digits, max_scaler):
    return max_scaler*torch.rand(digits.shape)
