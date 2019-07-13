import numpy as np
import scipy
from scipy.io import savemat
from .constants import RGB2YUV
from scipy.interpolate import interp2d


_RGB2YUV = RGB2YUV.cpu().data.numpy()


def ahd_demosaicking(mosaic, delta=1, sobel_sz=3, avg_sz=3):
    """Demosaicking using AHD algorithm.

        No median filtering, assume GRBG format.
        Args:
            delta: neighborhood size for calculating homogeneity.
            sobel_sz: size of sobel kernels.
            avg_sz: size of averaging kernel for homogeneity.
    """
    Yx = _demosaickX(mosaic)
    Yy = _demosaickY(mosaic)

    YxYUV = _rgb2YUV(Yx)
    YyYUV = _rgb2YUV(Yy)

    epsL, epsCsq = _adaptive_param(YxYUV, YyYUV, sobel_sz)

    Hx = _homogeniety(YxYUV, delta, epsL, epsCsq)
    Hy = _homogeniety(YyYUV, delta, epsL, epsCsq)

    Hx = _conv2(Hx, np.ones((avg_sz, avg_sz)) / float(avg_sz**2))
    Hy = _conv2(Hy, np.ones((avg_sz, avg_sz)) / float(avg_sz**2))

    mask = (Hx > Hy).astype('float')
    mask = np.expand_dims(mask, -1)
    output = mask * Yx + (1.0 - mask) * Yy
    return np.clip(output, 0.0, 1.0)


# https://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size/41065243#41065243
def _sobel_kernel(sz):
    if (sz % 2) == 0:
        raise ValueError("Kernel size must be odd ({} received)".format(sz))
    kernel = np.zeros((sz, sz))
    for i in range(sz):
        for j in range(sz):
            ii = i - (sz // 2)
            jj = j - (sz // 2)
            kernel[i, j] = ii / (ii**2 + jj**2) if ii != 0 else 0
    return kernel


def _interp2d(arr, new_sz):
    f = interp2d(x=np.linspace(0, 1, arr.shape[1]),
                 y=np.linspace(0, 1, arr.shape[0]),
                 z=arr)
    return f(np.linspace(0, 1, new_sz[1]), np.linspace(0, 1, new_sz[0]))


def _interp_kernel(m=5, n=3):
    # Duplicate row so it works with bilinear interpolation
    Hg = np.array([[-0.25, 0.5, 0.5, 0.5, -0.25],[-0.25, 0.5, 0.5, 0.5, -0.25]])
    Hr = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])
    if m != 5:
        Hg = _interp2d(Hg, (2, m))
    if n != 3:
        Hr = _interp2d(Hr, (n, n))
    Hg = Hg[0:1, :]
    Hg = Hg / np.sum(Hg[:])
    Hr = Hr / np.sum(Hr[:]) * 4
    return Hg, Hr


def _conv2(x, k):
    return scipy.ndimage.filters.convolve(x, k, mode='reflect')


def _demosaickX(X, transposed=False):
    Mr = np.zeros(X.shape)
    Mg = np.ones(X.shape)
    Mb = np.zeros(X.shape)

    Mr[0::2, 1::2] = 1.0
    Mb[1::2, 0::2] = 1.0
    Mg = Mg - Mr - Mb
    # Switch R and B (which got swapped when we transpose X).
    if transposed:
        Mr, Mb = Mb, Mr

    Hg, Hr = _interp_kernel(5, 3)
    G = Mg * X + (Mr + Mb) * _conv2(X, Hg)

    R = G + _conv2(Mr * (X - G), Hr)
    B = G + _conv2(Mb * (X - G), Hr)
    R = np.expand_dims(R, -1)
    G = np.expand_dims(G, -1)
    B = np.expand_dims(B, -1)
    return np.concatenate((R,G,B), axis=2)


def _demosaickY(X):
    X = X.T
    Y = _demosaickX(X, transposed=True)
    Y = np.swapaxes(Y, 0, 1)
    return Y


def _adaptive_param(X, Y, sz):
    sobel_y = _sobel_kernel(sz)
    sobel_x = sobel_y.T
    eL = np.minimum(abs(_conv2(X[:,:,0], sobel_x)),
                    abs(_conv2(Y[:,:,0], sobel_y)))
    eCsq = np.minimum(_conv2(X[:,:,1], sobel_x)**2 + _conv2(X[:,:,2], sobel_x)**2,
                      _conv2(Y[:,:,1], sobel_y)**2 + _conv2(Y[:,:,2], sobel_y)**2)
    return eL, eCsq


def _rgb2YUV(X):
    return np.einsum("ijk,lk->ijl", X, _RGB2YUV)


def _ballset(delta):
    index = int(np.ceil(delta))
    # initialize
    H = np.zeros((index*2+1, index*2+1, (index*2+1)**2))
    k = 0;
    for i in range(-index, index):
        for j in range(-index,index):
            if np.sqrt(i**2 + j**2) <= delta:
                # included
                H[index+i, index+j, k] = 1
                k = k + 1
    H = H[:,:,:k];
    return H


def _homogeniety(X, delta, epsL, epsC_sq):
    H = _ballset(delta);

    K = np.zeros(X.shape[:2])

    for i in range(H.shape[-1]):
        # level set
        L = abs(_conv2(X[:,:,0], H[:,:,i]) - X[:,:,0]) <= epsL
        # color set
        C = ((_conv2(X[:,:,1], H[:,:,i]) - X[:,:,1])**2 + \
             (_conv2(X[:,:,2], H[:,:,i]) - X[:,:,2])**2) <= epsC_sq;
        # metric neighborhood
        U = C * L
        # homogeneity
        K = K + U
    return K
