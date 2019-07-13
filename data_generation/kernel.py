import torch


def gausskern1d(sig, sz=None):
    """ 1D Gaussian kernel.

    Args:
        sz: kernel size.
        sig: stdev of the kernel
    """
    if sz is None:
        sz = int(2*int(sig) + 1)
        sz = max(sz, 3)
    half_sz = int(sz / 2)
    neg_half_sz = half_sz - sz + 1
    neg_half_sz = float(neg_half_sz)
    half_sz = float(half_sz)
    x = torch.linspace(neg_half_sz, half_sz, int(sz)) / sig
    x = x ** 2
    kern = torch.exp(-x/2.0)
    kern = kern / kern.sum()
    return kern

def gausskern2d(sz_x, sig_x, sz_y=None, sig_y=None):
    """Returns a 2D Gaussian kernel array.

        Modified from https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy

    Args:
        sz_{x,y}: kernel size.
        sig_{x,y}: stdev of kernel in each direction
    """
    if sz_y is None:
        sz_y = sz_x
    if sig_y is None:
        sig_y = sig_x

    kern1d_x = gausskern1d(sz_x, sig_x)
    kern1d_y = gausskern1d(sz_y, sig_y)
    kernel_raw = torch.einsum('i,j->ij', kern1d_x, kern1d_y)
    # This einsum is equivalent to outer product (no repeated indices).
    # For future reference
    # kernel_raw = np.sqrt(np.einsum('ij,k', kernel_raw, kern_r))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel
