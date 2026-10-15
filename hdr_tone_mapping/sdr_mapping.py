"""Shared luminance-only SDR ACES Filmic for both experiments."""
import numpy as np


def encode_numpy(rgb, matrix, exposure_nits=100):
    """BT.2020 nits -> Y-only ACES -> BT.709/sRGB, with 100-nit white."""
    if exposure_nits <= 0:
        raise ValueError('exposure_nits must be positive')
    y = rgb @ np.array([.2627, .6780, .0593])
    x = np.maximum(y, 0)/exposure_nits
    target = 100*np.clip(x*(2.51*x+.03)/(x*(2.43*x+.59)+.14), 0, 1)
    gain = np.divide(target, y, out=np.zeros_like(target), where=y > 0)
    relative = ((rgb*gain[..., None]) @ matrix.T)/100
    clipped = int(np.count_nonzero(np.any((relative < 0) | (relative > 1), axis=-1)))
    linear = np.clip(relative, 0, 1)
    encoded = np.where(linear <= .0031308, 12.92*linear, 1.055*linear**(1/2.4)-.055)
    return encoded, clipped


def encode_tensor(rgb, y, matrix, exposure_nits=100):
    """Same operator on PyTorch tensors; caller owns inference mode/device."""
    import torch
    if exposure_nits <= 0:
        raise ValueError('exposure_nits must be positive')
    x = y.clamp_min(0)/exposure_nits
    target = 100*(x*(2.51*x+.03)/(x*(2.43*x+.59)+.14)).clamp(0, 1)
    gain = torch.where(y > 0, target/y.clamp_min(torch.finfo(y.dtype).tiny), 0)
    relative = ((rgb*gain[..., None]) @ matrix.T)/100
    clipped = int(((relative < 0) | (relative > 1)).any(dim=-1).sum().item())
    linear = relative.clamp(0, 1)
    encoded = torch.where(linear <= .0031308, 12.92*linear, 1.055*linear.pow(1/2.4)-.055)
    return encoded, clipped
