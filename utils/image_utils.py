import numpy as np
import torch


def center_crop_tensor(tensor, w, h):
    tw = tensor.size(-1)
    th = tensor.size(-2)
    if tw < w or th < h:
        raise RuntimeError("Crop size is larger than image size.")
    h0 = int((th - h) / 2)
    w0 = int((tw - w) / 2)
    h1 = h0 + h
    w1 = w0 + w
    return tensor[..., h0:h1, w0:w1]


def bayer_crop_tensor(tensor, w, h, mode="random"):
    """Crop that preserves Bayer phase"""
    tw = tensor.size(-1)
    th = tensor.size(-2)
    if tw < w or th < h:
        raise RuntimeError("Crop size ({}) is larger than image size ({})." \
                            .format((w, h), (tw, th)))
    if mode == "random":
        h0 = np.random.choice(th + 1 - h)
        w0 = np.random.choice(tw + 1 - w)
    elif mode == "center":
        h0 = int((th - h) / 2)
        w0 = int((tw - w) / 2)
    else:
        raise ValueError("Bayer crop: unrecognized mode ({}). Must be 'random' or 'center'.".format(mode))
    # make sure start index is divisible by 2
    h0 = h0 - (h0 % 2)
    w0 = w0 - (w0 % 2)
    h1 = h0 + h
    w1 = w0 + w
    return tensor[..., h0:h1, w0:w1]


def random_crop_tensor(tensor, w, h):
    tw = tensor.size(-1)
    th = tensor.size(-2)
    if tw < w or th < h:
        raise RuntimeError("Crop size is larger than image size.")
    h0 = np.random.randint(th - h)
    w0 = np.random.randint(tw - w)
    h1 = h0 + h
    w1 = w0 + w
    return tensor[..., h0:h1, w0:w1]


def check_nan_tensor(x):
    return torch.isnan(x).any()
