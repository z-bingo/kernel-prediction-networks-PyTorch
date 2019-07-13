""" Utilities functions.
"""
import numbers
import numpy as np
import torch
from torch import FloatTensor


def random_crop(im, num_patches, w, h=None):
    h = w if h is None else h
    nw = im.size(-1) - w
    nh = im.size(-2) - h
    if nw < 0 or nh < 0:
        raise RuntimeError("Image is to small {} for the desired size {}". \
                                format((im.size(-1), im.size(-2)), (w, h))
                          )

    idx_w = np.random.choice(nw + 1, size=num_patches)
    idx_h = np.random.choice(nh + 1, size=num_patches)

    result = []
    for i in range(num_patches):
        result.append(im[...,
                         idx_h[i]:(idx_h[i]+h),
                         idx_w[i]:(idx_w[i]+w)])
    return result


def expand_to_4d_channel(arr):
    """ Expand Scalar or 1D dimension to 4D

    Assumes that a 1D list represent the channel dimension (2nd dim).

    Args:
        arr: A scalar or 1D tensor to be expanded to 4D
    """
    # for scalar and 1D tensor, add batch dimensions.
    while len(arr.size()) < 2:
        arr = arr.unsqueeze(0)
    # regain spatial dimension
    while len(arr.size()) < 4:
        arr = arr.unsqueeze(-1)
    return arr


def expand_to_4d_batch(arr):
    """ Expand Scalar or 1D dimension to 4D

    Assumes that a 1D list represent the batch dimension (1st dim).

    Args:
        arr: A scalar or 1D tensor to be expanded to 4D
    """
    # regain spatial dimension and channel dimension
    while len(arr.size()) < 4:
        arr = arr.unsqueeze(-1)
    return arr


def is_number(a):
    return isinstance(a, numbers.Number)


def python_to_tensor(a):
    if isinstance(a, numbers.Number):
        return FloatTensor([a])
    return a


def number_to_list(a):
    if isinstance(a, numbers.Number):
        a = [a]
    return a


def cuda_like(arr, src):
    """ Move arr on to GPU/CPU like src
    """
    if src.is_cuda:
        return arr.cuda()
    else:
        return arr.cpu()


def mosaick_multiply(mult, im, mosaick_pattern):
    """ mosaick pattern-aware multiply.

    Args:
        mult: n-list of multiplier, where n is number of image channel.
            A batch dimension is optional.
        im: tensor of size n_batch x n_channel x width x height.
        mosaick_pattern: None or string indicating the mosaick pattern.
    """
    if mosaick_pattern is None:
        return im * expand_to_4d_channel(mult)
    elif mosaick_pattern == "bayer":
        # Assume GRGB format.
        mult = expand_to_4d_channel(mult)

        h, w = im.size(2), im.size(3)
        x = torch.arange(w).unsqueeze(0).expand(h, -1)
        y = torch.arange(h).unsqueeze(-1).expand(-1, w)
        x = x.unsqueeze(0).unsqueeze(0)
        y = y.unsqueeze(0).unsqueeze(0)

        if im.is_cuda:
            x = x.cuda()
            y = y.cuda()

        odd_x = torch.fmod(x, 2)
        odd_y = torch.fmod(y, 2)

        is_green = odd_x == odd_y
        is_red = odd_x * (1.0 - odd_y)
        is_blue = (1.0 - odd_x) * odd_y

        mult = mult.expand(-1, 3, -1, -1)

        return im * mult[:, 0:1, ...] * is_red.float() + \
            im * mult[:, 1:2, ...] * is_green.float() + \
            im * mult[:, 2:3, ...] * is_blue.float()
    else:
        raise ValueError("Mosaick pattern, {}, not supported." \
                         .format(mosaick_pattern))
