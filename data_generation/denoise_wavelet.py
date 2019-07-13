# coding: utf-8
# Modified from skimage's wavelet.
# https://github.com/scikit-image/scikit-image/blob/f0d48db4c246989182aa01c837d04903bc2330ae/skimage/restoration/_denoise.py
import scipy.stats
import numpy as np
from math import ceil
import warnings
import pywt
import skimage.color as color
from skimage import img_as_float
import numbers


def warn(msg):
    warnings.warn(msg)


def _bayes_thresh(details, var):
    """BayesShrink threshold for a zero-mean details coeff array."""
    # Equivalent to:  dvar = np.var(details) for 0-mean details array
    dvar = np.mean(details*details)
    eps = np.finfo(details.dtype).eps
    thresh = var / np.sqrt(max(dvar - var, eps))
    return thresh


def _universal_thresh(img, sigma):
    """ Universal threshold used by the VisuShrink method """
    return sigma*np.sqrt(2*np.log(img.size))


def _sigma_est_dwt(detail_coeffs, distribution='Gaussian'):
    """Calculate the robust median estimator of the noise standard deviation.

    Parameters
    ----------
    detail_coeffs : ndarray
        The detail coefficients corresponding to the discrete wavelet
        transform of an image.
    distribution : str
        The underlying noise distribution.

    Returns
    -------
    sigma : float
        The estimated noise standard deviation (see section 4.2 of [1]_).

    References
    ----------
    .. [1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
       by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
       DOI:10.1093/biomet/81.3.425
    """
    # Consider regions with detail coefficients exactly zero to be masked out
    detail_coeffs = detail_coeffs[np.nonzero(detail_coeffs)]

    if distribution.lower() == 'gaussian':
        # 75th quantile of the underlying, symmetric noise distribution
        denom = scipy.stats.norm.ppf(0.75)
        sigma = np.median(np.abs(detail_coeffs)) / denom
    else:
        raise ValueError("Only Gaussian noise estimation is currently "
                         "supported")
    return sigma


def _wavelet_threshold(image, wavelet, method=None, threshold=None,
                       sigma=None, mode='soft', wavelet_levels=None):
    """Perform wavelet thresholding.

    Parameters
    ----------
    image : ndarray (2d or 3d) of ints, uints or floats
        Input data to be denoised. `image` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.
    wavelet : string
        The type of wavelet to perform. Can be any of the options
        pywt.wavelist outputs. For example, this may be any of ``{db1, db2,
        db3, db4, haar}``.
    method : {'BayesShrink', 'VisuShrink'}, optional
        Thresholding method to be used. The currently supported methods are
        "BayesShrink" [1]_ and "VisuShrink" [2]_. If it is set to None, a
        user-specified ``threshold`` must be supplied instead.
    threshold : float, optional
        The thresholding value to apply during wavelet coefficient
        thresholding. The default value (None) uses the selected ``method`` to
        estimate appropriate threshold(s) for noise removal.
    sigma : float, optional
        The standard deviation of the noise. The noise is estimated when sigma
        is None (the default) by the method in [2]_.
    mode : {'soft', 'hard'}, optional
        An optional argument to choose the type of denoising performed. It
        noted that choosing soft thresholding given additive noise finds the
        best approximation of the original image.
    wavelet_levels : int or None, optional
        The number of wavelet decomposition levels to use.  The default is
        three less than the maximum number of possible decomposition levels
        (see Notes below).

    Returns
    -------
    out : ndarray
        Denoised image.

    References
    ----------
    .. [1] Chang, S. Grace, Bin Yu, and Martin Vetterli. "Adaptive wavelet
           thresholding for image denoising and compression." Image Processing,
           IEEE Transactions on 9.9 (2000): 1532-1546.
           DOI: 10.1109/83.862633
    .. [2] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
           by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
           DOI: 10.1093/biomet/81.3.425

    """
    wavelet = pywt.Wavelet(wavelet)

    # original_extent is used to workaround PyWavelets issue #80
    # odd-sized input results in an image with 1 extra sample after waverecn
    original_extent = tuple(slice(s) for s in image.shape)

    # Determine the number of wavelet decomposition levels
    if wavelet_levels is None:
        # Determine the maximum number of possible levels for image
        dlen = wavelet.dec_len
        wavelet_levels = np.min(
            [pywt.dwt_max_level(s, dlen) for s in image.shape])

        # Skip coarsest wavelet scales (see Notes in docstring).
        wavelet_levels = max(wavelet_levels - 3, 1)

    coeffs = pywt.wavedecn(image, wavelet=wavelet, level=wavelet_levels)
    # Detail coefficients at each decomposition level
    dcoeffs = coeffs[1:]

    if sigma is None:
        # Estimate the noise via the method in [2]_
        detail_coeffs = dcoeffs[-1]['d' * image.ndim]
        sigma = _sigma_est_dwt(detail_coeffs, distribution='Gaussian')

    if method is not None and threshold is not None:
        warn(("Thresholding method {} selected.  The user-specified threshold "
              "will be ignored.").format(method))

    if threshold is None:
        var = sigma**2
        if method is None:
            raise ValueError(
                "If method is None, a threshold must be provided.")
        elif method == "BayesShrink":
            # The BayesShrink thresholds from [1]_ in docstring
            threshold = [{key: _bayes_thresh(level[key], var) for key in level}
                         for level in dcoeffs]
        elif method == "VisuShrink":
            # The VisuShrink thresholds from [2]_ in docstring
            threshold = _universal_thresh(image, sigma)
        else:
            raise ValueError("Unrecognized method: {}".format(method))
    elif isinstance(threshold, list) or isinstance(threshold, tuple):
        # convert to list in case it's a tuple
        threshold = list(threshold)
        # Supplement threshold if it is not long enough
        # higher level is lower-frequency details
        if len(threshold) < len(dcoeffs):
            threshold = threshold + ([threshold[-1]] * (len(dcoeffs) - len(threshold)))
        threshold = [{key: threshold[i] for key in level}
                         for i, level in enumerate(dcoeffs)]

    if np.isscalar(threshold):
        # A single threshold for all coefficient arrays
        denoised_detail = [{key: pywt.threshold(level[key],
                                                value=threshold,
                                                mode=mode) for key in level}
                           for level in dcoeffs]
    else:
        # Dict of unique threshold coefficients for each detail coeff. array
        denoised_detail = [{key: pywt.threshold(level[key],
                                                value=thresh[key],
                                                mode=mode) for key in level}
                           for thresh, level in zip(threshold, dcoeffs)]
    denoised_coeffs = [coeffs[0]] + denoised_detail
    return pywt.waverecn(denoised_coeffs, wavelet)[original_extent]


def denoise_wavelet(image, sigma=None, wavelet='db1', mode='soft',
                    wavelet_levels=None, multichannel=False,
                    convert2ycbcr=False, method='BayesShrink',
                    threshold=None
                   ):
    """Perform wavelet denoising on an image.

    Parameters
    ----------
    image : ndarray ([M[, N[, ...P]][, C]) of ints, uints or floats
        Input data to be denoised. `image` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.
    sigma : float or list, optional
        The noise standard deviation used when computing the wavelet detail
        coefficient threshold(s). When None (default), the noise standard
        deviation is estimated via the method in [2]_.
    wavelet : string, optional
        The type of wavelet to perform and can be any of the options
        ``pywt.wavelist`` outputs. The default is `'db1'`. For example,
        ``wavelet`` can be any of ``{'db2', 'haar', 'sym9'}`` and many more.
    mode : {'soft', 'hard'}, optional
        An optional argument to choose the type of denoising performed. It
        noted that choosing soft thresholding given additive noise finds the
        best approximation of the original image.
    wavelet_levels : int or None, optional
        The number of wavelet decomposition levels to use.  The default is
        three less than the maximum number of possible decomposition levels.
    multichannel : bool, optional
        Apply wavelet denoising separately for each channel (where channels
        correspond to the final axis of the array).
    convert2ycbcr : bool, optional
        If True and multichannel True, do the wavelet denoising in the YCbCr
        colorspace instead of the RGB color space. This typically results in
        better performance for RGB images.
    method : {'BayesShrink', 'VisuShrink'}, optional
        Thresholding method to be used. The currently supported methods are
        "BayesShrink" [1]_ and "VisuShrink" [2]_. Defaults to "BayesShrink".

    Returns
    -------
    out : ndarray
        Denoised image.

    Notes
    -----
    The wavelet domain is a sparse representation of the image, and can be
    thought of similarly to the frequency domain of the Fourier transform.
    Sparse representations have most values zero or near-zero and truly random
    noise is (usually) represented by many small values in the wavelet domain.
    Setting all values below some threshold to 0 reduces the noise in the
    image, but larger thresholds also decrease the detail present in the image.

    If the input is 3D, this function performs wavelet denoising on each color
    plane separately. The output image is clipped between either [-1, 1] and
    [0, 1] depending on the input image range.

    When YCbCr conversion is done, every color channel is scaled between 0
    and 1, and `sigma` values are applied to these scaled color channels.

    Many wavelet coefficient thresholding approaches have been proposed.  By
    default, ``denoise_wavelet`` applies BayesShrink, which is an adaptive
    thresholding method that computes separate thresholds for each wavelet
    sub-band as described in [1]_.

    If ``method == "VisuShrink"``, a single "universal threshold" is applied to
    all wavelet detail coefficients as described in [2]_.  This threshold
    is designed to remove all Gaussian noise at a given ``sigma`` with high
    probability, but tends to produce images that appear overly smooth.

    References
    ----------
    .. [1] Chang, S. Grace, Bin Yu, and Martin Vetterli. "Adaptive wavelet
           thresholding for image denoising and compression." Image Processing,
           IEEE Transactions on 9.9 (2000): 1532-1546.
           DOI: 10.1109/83.862633
    .. [2] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
           by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
           DOI: 10.1093/biomet/81.3.425

    Examples
    --------
    >>> from skimage import color, data
    >>> img = img_as_float(data.astronaut())
    >>> img = color.rgb2gray(img)
    >>> img += 0.1 * np.random.randn(*img.shape)
    >>> img = np.clip(img, 0, 1)
    >>> denoised_img = denoise_wavelet(img, sigma=0.1)

    """

    if method not in ["BayesShrink", "VisuShrink"]:
        raise ValueError(
            ('Invalid method: {}. The currently supported methods are '
             '"BayesShrink" and "VisuShrink"').format(method))

    image = img_as_float(image)

    if multichannel:
        if isinstance(sigma, numbers.Number) or sigma is None:
            sigma = [sigma] * image.shape[-1]

    if multichannel:
        if convert2ycbcr:
            out = color.rgb2ycbcr(image)
            for i in range(3):
                # renormalizing this color channel to live in [0, 1]
                min, max = out[..., i].min(), out[..., i].max()
                channel = out[..., i] - min
                channel /= max - min
                out[..., i] = denoise_wavelet(channel, wavelet=wavelet,
                                              method=method, sigma=sigma[i],
                                              mode=mode,
                                              wavelet_levels=wavelet_levels,
                                              threshold=threshold
                                             )

                out[..., i] = out[..., i] * (max - min)
                out[..., i] += min
            out = color.ycbcr2rgb(out)
        else:
            out = np.empty_like(image)
            for c in range(image.shape[-1]):
                out[..., c] = _wavelet_threshold(image[..., c],
                                                 wavelet=wavelet,
                                                 method=method,
                                                 sigma=sigma[c], mode=mode,
                                                 wavelet_levels=wavelet_levels,
                                                 threshold=threshold)
    else:
        out = _wavelet_threshold(image, wavelet=wavelet, method=method,
                                 sigma=sigma, mode=mode,
                                 wavelet_levels=wavelet_levels,
                                 threshold=threshold)

    clip_range = (-1, 1) if image.min() < 0 else (0, 1)
    return np.clip(out, *clip_range)


def estimate_sigma(image, average_sigmas=False, multichannel=False):
    """
    Robust wavelet-based estimator of the (Gaussian) noise standard deviation.

    Parameters
    ----------
    image : ndarray
        Image for which to estimate the noise standard deviation.
    average_sigmas : bool, optional
        If true, average the channel estimates of `sigma`.  Otherwise return
        a list of sigmas corresponding to each channel.
    multichannel : bool
        Estimate sigma separately for each channel.

    Returns
    -------
    sigma : float or list
        Estimated noise standard deviation(s).  If `multichannel` is True and
        `average_sigmas` is False, a separate noise estimate for each channel
        is returned.  Otherwise, the average of the individual channel
        estimates is returned.

    Notes
    -----
    This function assumes the noise follows a Gaussian distribution. The
    estimation algorithm is based on the median absolute deviation of the
    wavelet detail coefficients as described in section 4.2 of [1]_.

    References
    ----------
    .. [1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
       by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
       DOI:10.1093/biomet/81.3.425

    Examples
    --------
    >>> import skimage.data
    >>> from skimage import img_as_float
    >>> img = img_as_float(skimage.data.camera())
    >>> sigma = 0.1
    >>> img = img + sigma * np.random.standard_normal(img.shape)
    >>> sigma_hat = estimate_sigma(img, multichannel=False)
    """
    if multichannel:
        nchannels = image.shape[-1]
        sigmas = [estimate_sigma(
            image[..., c], multichannel=False) for c in range(nchannels)]
        if average_sigmas:
            sigmas = np.mean(sigmas)
        return sigmas
    elif image.shape[-1] <= 4:
        msg = ("image is size {0} on the last axis, but multichannel is "
               "False.  If this is a color image, please set multichannel "
               "to True for proper noise estimation.")
        warn(msg.format(image.shape[-1]))
    coeffs = pywt.dwtn(image, wavelet='db2')
    detail_coeffs = coeffs['d' * image.ndim]
    return _sigma_est_dwt(detail_coeffs, distribution='Gaussian')
