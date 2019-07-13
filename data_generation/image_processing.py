import torch
import torch.nn as nn
from torch import FloatTensor, IntTensor
# For drawing motion blur kernel.
import numpy as np
import cv2
import scipy
import functools
import math

from .data_utils import mosaick_multiply, expand_to_4d_batch
from .data_utils import python_to_tensor, cuda_like, number_to_list, is_number
from .kernel import gausskern1d, gausskern2d
from .constants import xyz_color_matching, XYZ2sRGB
from .constants import RGB2YUV, YUV2RGB
from .constants import DCT_coeff
from .constants import photoshop_jpeg_quantization_lum
from .constants import photoshop_jpeg_quantization_chrom
from .constants import photoshop_chroma_subsampling
from .ahd_demosaicking import ahd_demosaicking
from utils.image_utils import check_nan_tensor
import skimage
from .denoise_wavelet import denoise_wavelet as sk_denoise_wavelet

try:
    from halide.gradient_apps.gapps import functions as halide_funcs

    HAS_HALIDE = True
except:
    HAS_HALIDE = False

DEBUG = False


def _has_halide():
    return HAS_HALIDE


# TODO: Check if I need to set required_grad properly on all constant tensors.
class IdentityModule(nn.Module):
    """Dummy Class for testing."""

    def __init__(self):
        super().__init__()

    def forward(self, image):
        return image.copy()


# Halide实现
# Cellphone Image Processing
class DenoisingBilateral(nn.Module):
    # TODO: support batch
    # TODO: support GPU.
    def __init__(self,
                 sigma_s,
                 sigma_r,
                 color_sigma_ratio=5,
                 filter_lum=True,
                 filter_chrom=True,
                 n_iter=1,
                 guide_transform=None,
                 _bp=0.004,
                 color_range_ratio=1):
        """ Apply Gaussian bilateral filter to denoise image.

        Args:
            sigma_s: stdev in spatial dimension.
            sigma_r: stdev in the range dimension.
            color_sigma_ratio: multiplier for spatial sigma for filtering
                chrominance.
            filter_lum: whether or not to filter luminance (useful if want to
                filter chrominance only).
            filter_chrom: same as filter_lum but for chrominance.
            n_iter: number of times to apply this filter.
            guide_transform: transformation to apply to the guide map. Must be
                'sqrt', 'log', None, or a number. If a number, this is use as
                the exponent to transform the guide according to power law.
            _bp: Black point for log transform. This is used to prevent taking
                log of zeros or negative numbers. Must be positive.
            color_range_ratio: multiplier for range sigma for filtering
                chrominance.
        """
        super().__init__()
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r
        self.color_sigma_ratio = color_sigma_ratio
        self.color_range_ratio = color_range_ratio
        self.rgb2yuv = ColorSpaceConversionMatrix(RGB2YUV)
        self.yuv2rgb = ColorSpaceConversionMatrix(YUV2RGB)
        self.filter_lum = filter_lum
        self.filter_chrom = filter_chrom
        self.n_iter = n_iter
        self.guide_transform = guide_transform
        self._bp = _bp
        if self.guide_transform not in ['sqrt', 'log', None] and \
                not (is_number(self.guide_transform)):
            raise ValueError('Invalid guide transformation received: {}'.format(guide_transform))
        if self.guide_transform == 'sqrt':
            self.guide_transform = 0.5

    def forward(self, image):
        if not _has_halide():
            raise RuntimeError("Need halide in order to run this")
        if DEBUG and check_nan_tensor(image):
            print("Denoising input has NAN!")
        self._filter_s = FloatTensor(gausskern1d(self.sigma_s))
        self._filter_s_2 = FloatTensor(gausskern1d(3.0 * self.sigma_s))
        self._filter_s_color = FloatTensor(gausskern1d(self.color_sigma_ratio * self.sigma_s))
        self._filter_r = FloatTensor(gausskern1d(self.sigma_r))
        self._filter_r_color = FloatTensor(gausskern1d(self.sigma_r * self.color_range_ratio))
        self._filter_s = cuda_like(self._filter_s, image)
        self._filter_s_2 = cuda_like(self._filter_s_2, image)
        self._filter_s_color = cuda_like(self._filter_s_color, image)
        self._filter_r = cuda_like(self._filter_r, image)
        yuv = self.rgb2yuv(image)
        for i in range(self.n_iter):
            yuv = self._forward(yuv)
        output = self.yuv2rgb(yuv)
        if DEBUG and check_nan_tensor(output):
            print("Denoising output has NAN!")
        return output

    def _forward(self, yuv):
        lum = yuv[:, 0:1, ...]
        guide = lum[:, 0, ...]
        if is_number(self.guide_transform):
            guide = self._gamma_compression(guide, self.guide_transform)
        elif self.guide_transform == 'log':
            guide = self._log_compression(guide, self._bp)
        guide = torch.clamp(guide, 0.0, 1.0)

        out_yuv = yuv.clone()
        if self.filter_lum:
            out_lum = halide_funcs.BilateralGrid.apply(lum,
                                                       guide,
                                                       self._filter_s,
                                                       self._filter_r)
            out_yuv[:, 0:1, ...] = out_lum

        if self.filter_chrom:
            out_yuv[:, 1:3, ...] = halide_funcs.BilateralGrid.apply(yuv[:, 1:3, ...],
                                                                    out_yuv[:, 0, ...],
                                                                    self._filter_s_color,
                                                                    self._filter_r_color)
        return out_yuv

    @staticmethod
    def _gamma_compression(lum, gamma):
        return torch.pow(torch.clamp(lum, 0), gamma)

    @staticmethod
    def _undo_gamma_compression(lum, gamma):
        return torch.pow(torch.clamp(lum, 0), 1.0 / gamma)

    @staticmethod
    def _log_compression(lum, bp):
        # Just clamp
        log_bp = np.log(bp)
        lum = torch.log(torch.clamp(lum, bp))
        lum = torch.clamp((lum - log_bp) / (-log_bp), 0, 1)
        return lum

    @staticmethod
    def _undo_log_compression(lum, bp):
        # Add and rescale
        log_bp = np.log(bp)
        log_1_bp = np.log(1.0 + bp)
        lum = (lum * (log_1_bp - log_bp)) + log_bp
        lum = (torch.exp(lum) - bp)
        return lum


# 双边滤波非差分实现  不使用Halide
class DenoisingSKImageBilateralNonDifferentiable(DenoisingBilateral):

    def forward(self, image):
        if DEBUG and check_nan_tensor(image):
            print("Denoising input has NAN!")
        yuv = self.rgb2yuv(image)

        for i in range(self.n_iter):
            yuv = self._forward(yuv)
        output = self.yuv2rgb(yuv)
        if DEBUG and check_nan_tensor(output):
            print("Denoising output has NAN!")
        return output

    def _forward(self, yuv):
        lum = yuv[:, 0:1, ...]
        lum = torch.clamp(lum, 0, 1)
        out_yuv = yuv.clone()
        # This is use to convert sigma_r so that it is in the same range as
        # Halide's bilateral grid
        HALIDE_RANGE_GRID = 32.0
        skbilateral = skimage.restoration.denoise_bilateral
        if self.filter_lum:
            # skimage's bilateral filter uses the luminance as the guide.
            if is_number(self.guide_transform):
                lum = self._gamma_compression(lum, self.guide_transform)
            elif self.guide_transform == 'log':
                lum = self._log_compression(lum, self._bp)
            lum_ = lum.cpu().permute(0, 2, 3, 1).data.numpy().astype('float32')
            lum_ = lum_[:, :, :, 0]
            # Filter each image in the batch
            for i in range(lum_.shape[0]):
                # lum_[i, ...] = skbilateral(lum_[i, ...],
                #                            sigma_color=self.sigma_r / HALIDE_RANGE_GRID,
                #                            sigma_spatial=self.sigma_s,
                #                            multichannel=False,
                #                            mode="reflect")
                win_sz = max(5, 2 * math.ceil(3 * self.sigma_s) + 1)
                lum_[i, ...] = cv2.bilateralFilter(lum_[i, ...],
                                                   d=win_sz,
                                                   sigmaColor=self.sigma_r / HALIDE_RANGE_GRID,
                                                   sigmaSpace=self.sigma_s,
                                                   borderType=cv2.BORDER_REFLECT)
            lum_ = FloatTensor(lum_).unsqueeze(-1).permute(0, 3, 1, 2)
            out_lum = cuda_like(lum_, lum)
            # Undo guide transformation
            if is_number(self.guide_transform):
                out_lum = self._undo_gamma_compression(out_lum, self.guide_transform)
            elif self.guide_transform == 'log':
                out_lum = self._undo_log_compression(out_lum, self._bp)
            out_lum = torch.clamp(out_lum, 0.0, 1.0)
            out_yuv[:, 0:1, ...] = out_lum

        # Filter chrominance.
        if self.filter_chrom:
            chrom = yuv[:, 1:3, ...]
            chrom = torch.clamp((chrom + 1) * 0.5, 0.0, 1.0)
            chrom_ = chrom.cpu().permute(0, 2, 3, 1).data.numpy().astype('float32')
            # Filter each image in the batch
            for i in range(chrom_.shape[0]):
                for j in range(2):
                    # chrom_[i, :, :, j] = skbilateral(chrom_[i, :, :, j],
                    #                                 sigma_color=self.sigma_r / HALIDE_RANGE_GRID * self.color_range_ratio,
                    #                                 sigma_spatial=(self.sigma_s * self.color_sigma_ratio),
                    #                                 multichannel=False,
                    #                                 mode="reflect")
                    win_sz = max(5, 2 * math.ceil(3 * self.sigma_s * self.color_sigma_ratio) + 1)
                    chrom_[i, :, :, j] = cv2.bilateralFilter(chrom_[i, :, :, j],
                                                             d=win_sz,
                                                             sigmaColor=self.sigma_r / HALIDE_RANGE_GRID * self.color_range_ratio,
                                                             sigmaSpace=self.sigma_s * self.color_sigma_ratio,
                                                             borderType=cv2.BORDER_REFLECT)
            # Convert back to PyTorch tensor.
            chrom_ = FloatTensor(chrom_).permute(0, 3, 1, 2)
            out_chrom = cuda_like(chrom_, chrom)
            out_chrom = 2.0 * out_chrom - 1.0
            out_yuv[:, 1:3, ...] = out_chrom
        return out_yuv


class DenoisingWaveletNonDifferentiable(DenoisingSKImageBilateralNonDifferentiable):

    def __init__(self, **kwargs):
        """ HACK: this function repurpose input for bilateral filters for
                different things.

            sigma_s --> Thresholding method. Can be string of numerical flags.
            color_sigma_ratio --> String indicating wavelet family (see skimage's documentation for detail).
            n_iter --> levels of wavelets.
            _bp --> wavelet threshold.
        """
        super().__init__(**kwargs)
        if is_number(self.sigma_s):
            self.method = "BayesShrink" if self.sigma_s < 1 else "VisuShrink"
        else:
            self.method = self.sigma_s
        if is_number(self.color_sigma_ratio):
            raise ValueError("Wavelet denoising uses color_sigma_ratio to be"
                             " string indicating wavelet family to use. "
                             "{} received.".format(self.color_sigma_ratio))
        self.wavelet_family = self.color_sigma_ratio
        self.wavelet_levels = self.n_iter
        self.n_iter = 1
        self.wavelet_threshold = self._bp

    def _forward(self, yuv):
        lum = yuv[:, 0:1, ...]
        out_yuv = yuv.clone()
        # this is use to convert sigma_r so that it is in the same range as Halide's bilateral grid
        # HALIDE_RANGE_GRID = 32.0
        if self.filter_lum:
            if is_number(self.guide_transform):
                lum = self._gamma_compression(lum, self.guide_transform)
            elif self.guide_transform == 'log':
                lum = self._log_compression(lum, self._bp)
            lum_ = lum.cpu().permute(0, 2, 3, 1).data.numpy().astype('float64')
            lum_ = lum_[:, :, :, 0]
            for i in range(lum_.shape[0]):
                lum_[i, ...] = sk_denoise_wavelet(lum_[i, ...],
                                                  sigma=self.sigma_r,
                                                  method=self.method,
                                                  wavelet=self.wavelet_family,
                                                  wavelet_levels=self.wavelet_levels,
                                                  threshold=self.wavelet_threshold,
                                                  mode="soft")
            lum_ = FloatTensor(lum_).unsqueeze(-1).permute(0, 3, 1, 2)
            out_lum = cuda_like(lum_, lum)
            if is_number(self.guide_transform):
                out_lum = self._undo_gamma_compression(out_lum, self.guide_transform)
            elif self.guide_transform == 'log':
                out_lum = self._undo_log_compression(out_lum, self._bp)
            out_lum = torch.clamp(out_lum, 0.0, 1.0)
            out_yuv[:, 0:1, ...] = out_lum

        if self.filter_chrom:
            chrom = yuv[:, 1:3, ...]
            chrom = torch.clamp((chrom + 1) * 0.5, 0.0, 1.0)
            chrom_ = chrom.cpu().permute(0, 2, 3, 1).data.numpy().astype('float64')
            for i in range(chrom_.shape[0]):
                chrom_[i, ...] = sk_denoise_wavelet(chrom_[i, ...],
                                                    method=self.method,
                                                    wavelet=self.wavelet_family,
                                                    wavelet_levels=self.wavelet_levels,
                                                    threshold=self.wavelet_threshold,
                                                    mode="soft")
            chrom_ = FloatTensor(chrom_).permute(0, 3, 1, 2)
            out_chrom = cuda_like(chrom_, chrom)
            out_chrom = 2.0 * out_chrom - 1.0
            out_yuv[:, 1:3, ...] = out_chrom
        return out_yuv


class DenoisingMedianNonDifferentiable(nn.Module):

    def __init__(self,
                 neighbor_sz,
                 color_sigma_ratio=5,
                 filter_lum=True,
                 filter_chrom=True,
                 n_iter=1):
        """ Apply Median Filtering
        """
        super().__init__()
        self.rgb2yuv = ColorSpaceConversionMatrix(RGB2YUV)
        self.yuv2rgb = ColorSpaceConversionMatrix(YUV2RGB)
        self.filter_lum = filter_lum
        self.filter_chrom = filter_chrom
        self.n_iter = n_iter
        self.lum_median = MedianFilterNonDifferentiable(neighbor_sz)
        if is_number(neighbor_sz):
            self.chrom_median = MedianFilterNonDifferentiable(int(neighbor_sz * color_sigma_ratio))
        else:
            if DEBUG and color_sigma_ratio != 1:
                print("Warning: ignoring color_sigma_ratio because neighbor_sz is not a number.")
            self.chrom_median = self.lum_median

    def forward(self, image):
        if DEBUG and check_nan_tensor(image):
            print("Denoising input has NAN!")
        yuv = self.rgb2yuv(image)
        for i in range(self.n_iter):
            yuv = self._forward(yuv)
        output = self.yuv2rgb(yuv)
        if DEBUG and check_nan_tensor(output):
            print("Denoising output has NAN!")
        return output

    def _forward(self, yuv):
        lum = yuv[:, 0:1, ...]
        out_yuv = yuv.clone()
        if self.filter_lum:
            out_lum = self.lum_median(lum)
            out_yuv[:, 0:1, ...] = torch.clamp(out_lum, 0.0, 1.0)
        if self.filter_chrom:
            out_yuv[:, 1:3, ...] = self.chrom_median(yuv[:, 1:3, ...])
        return out_yuv


class PytorchResizing(nn.Module):

    def __init__(self,
                 resizing_factor=None,
                 new_size=None,
                 mode='bilinear'):
        """ Bilinear interpolation for resizing.

        *** No Pre-filtering is applied!

        Args:
            resizing_factor: factors to resize image with. This or new_size
                must be specified.
            new_size: new image size (width, height) to resize to.
            mode: "bilinear", "area", "nearest". See nn.functional.interpolate
                for more detail.
        """
        super().__init__()
        if (new_size is None) == (resizing_factor is None):
            raise ValueError("Must specified exactly one of new_size ({})"
                             " or resizing_factor ({}).".format(new_size,
                                                                resizing_factor)
                             )
        self.resizing_factor = resizing_factor
        self.new_size = new_size
        self.mode = mode

    def forward(self, image):
        return nn.functional.interpolate(image,
                                         self.new_size,
                                         self.resizing_factor,
                                         mode=self.mode)


class MedianFilterNonDifferentiable(nn.Module):

    def __init__(self, filter_sz):
        super().__init__()
        if is_number(filter_sz):
            self.filter_sz = filter_sz
            self.footprint = None
        else:
            self.filter_sz = None
            self.footprint = filter_sz

    def forward(self, image):
        image_ = image.cpu().data.numpy()
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image_[i, j, ...] = scipy.ndimage.filters.median_filter(image_[i, j, ...], size=self.filter_sz,
                                                                        footprint=self.footprint)
        image_ = FloatTensor(image_)
        return cuda_like(image_, image)


class BicubicResizing(nn.Module):

    def __init__(self,
                 resizing_factor=None,
                 new_size=None,
                 B=1.0, C=0.0):
        """ Bicubic interpolation for resizing.

        *** No Pre-filtering is applied!

        Args:
            resizing_factor: factors to resize image with. This or new_size
                must be specified.
            new_size: new image size (width, height) to resize to.
            B, C: parameters of the spline (refer to Mitchell's SIGGRAPH'88 paper).
                Default is (1, 0) which makes this a B-spline.
        """
        super().__init__()
        if (new_size is None) == (resizing_factor is None):
            raise ValueError("Must specified exactly one of new_size ({})"
                             " or resizing_factor ({}).".format(new_size,
                                                                resizing_factor)
                             )
        self.resizing_factor = resizing_factor
        self.new_size = new_size
        self.B, self.C = B, C
        # The halide backend still needs debuging.
        raise NotImplementedError

    def forward(self, image):
        if self.resizing_factor is not None:
            sz = list(image.size())
            new_W = int(self.resizing_factor * sz[-1])
            new_H = int(self.resizing_factor * sz[-2])
            if new_W < 1 or new_H < 1:
                raise ValueError("Image to small that new size is zeros "
                                 "(w, h = {}, {})".format(new_W, new_H))
        else:
            new_W, new_H = int(self.new_size[0]), int(self.new_size[1])

        output = halide_funcs.BicubicResizing.apply(image,
                                                    new_W, new_H,
                                                    self.B, self.C)
        return output


class Unsharpen(nn.Module):

    def __init__(self, amount, radius, threshold, blur_filter_sz=None):
        """Unsharp an image.

        This doesn't support batching because GaussianBlur doesn't.

        Args:
            amount: (float) amount of sharpening to apply.
            radius: (float) radius of blur for the mask in pixel.
            threshold: (float) minimum brightness diff to operate on (on 0-255 scale)

        """
        super().__init__()
        self.amount = amount
        self.radius = radius
        self.threshold = threshold
        # if not specified, set it to twice the radius.
        if blur_filter_sz is None:
            self.filter_size = radius * 2
        else:
            self.filter_size = blur_filter_sz
        self.blur = GaussianBlur(self.radius,
                                 sz_x=self.filter_size,
                                 sz_y=self.filter_size)

    def forward(self, image):
        # Create unsharp mask
        unsharp_mask = image - self.blur(image)
        # Apply threshold
        unsharp_mask = unsharp_mask * (torch.abs(unsharp_mask) > (self.threshold / 255)).float()
        return image + unsharp_mask * self.amount


# Demosaicking
class NaiveDemosaicking(nn.Module):
    # TODO: Support GPU. Having host_dirty() exception now.
    def __init__(self, use_median_filter=True, n_iter=3, **kwargs):
        """
        Args:
            use_median_filter: whether or not to apply median filter on chrominance/luminance
            n_iter: number of times to apply median filters.
        """
        super().__init__()
        if use_median_filter:
            # Same footprint as in the original AHD algorithm.
            RB_footprint = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            G_footprint = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
            self.median_RB = MedianFilterNonDifferentiable(RB_footprint)
            self.median_G = MedianFilterNonDifferentiable(G_footprint)
        self.use_median_filter = use_median_filter
        self.n_iter = n_iter
        if _has_halide():
            self.demosaicker = halide_funcs.NaiveDemosaick.apply

    def forward(self, image):
        demosaicked = self.demosaicker(image)
        if self.use_median_filter:
            demosaicked_ = demosaicked.cpu()
            # repeat 3 times
            for i in range(self.n_iter):
                # follow AHD paper:
                # https://www.photoactivity.com/Pagine/Articoli/006NewDCRaw/hirakawa03adaptive.pdf
                R = demosaicked_[:, 0:1, ...].clone()
                G = demosaicked_[:, 1:2, ...].clone()
                B = demosaicked_[:, 2:3, ...].clone()
                R = self.median_RB(R - G) + G
                B = self.median_RB(B - G) + G
                G = 0.5 * (self.median_G(G - R) + \
                           self.median_G(G - B) + \
                           R + B)
                demosaicked_[:, 0:1, ...] = R
                demosaicked_[:, 1:2, ...] = G
                demosaicked_[:, 2:3, ...] = B
            demosaicked = cuda_like(demosaicked_, demosaicked)
        return demosaicked


class AHDDemosaickingNonDifferentiable(NaiveDemosaicking):
    # TODO: Convert Numpy to Pytorch
    def __init__(self, use_median_filter=True, n_iter=3, delta=2, sobel_sz=3, avg_sz=3):
        super().__init__(use_median_filter, n_iter)

        # print("Using AHD Non-differentiable")
        def ahd_demosaicker(image):
            image_ = image.cpu().permute(0, 2, 3, 1).squeeze(-1).data.numpy()
            output = []
            for i in range(image_.shape[0]):
                output.append(FloatTensor(ahd_demosaicking(image_[i, ...], delta, sobel_sz, avg_sz)).unsqueeze(0))
            output = cuda_like(torch.cat(output, dim=0).permute(0, 3, 1, 2), image)
            return output

        self.demosaicker = ahd_demosaicker


class BayerMosaicking(nn.Module):
    """ Turn 3-channel image into GRGB Bayer.
    """

    def forward(self, image):
        # Compute Meshgrid.
        # Tensors are batch x channels x height x width
        h, w = image.size(2), image.size(3)
        x = torch.arange(w).unsqueeze(0).expand(h, -1)
        y = torch.arange(h).unsqueeze(-1).expand(-1, w)
        x = x.unsqueeze(0).unsqueeze(0)
        y = y.unsqueeze(0).unsqueeze(0)

        if image.is_cuda:
            x = x.cuda()
            y = y.cuda()

        odd_x = torch.fmod(x, 2)
        odd_y = torch.fmod(y, 2)

        is_green = odd_x == odd_y
        is_red = odd_x * (1.0 - odd_y)
        is_blue = (1.0 - odd_x) * odd_y

        return image[:, 0:1, :, :] * is_red.float() + \
               image[:, 1:2, :, :] * is_green.float() + \
               image[:, 2:3, :, :] * is_blue.float()


# Color
class WhiteBalance(nn.Module):

    def __init__(self, scaling, mosaick_pattern=None):
        """ Perform white balance with a scaling factor.

        Args:
            scaling: Tensor of size [channels] for scaling each channel
                of the image. Batch dimension is optional.
            mosaick_pattern: mosaick pattern of the input image.
        """
        super().__init__()
        self.scaling = scaling
        self.mosaick_pattern = mosaick_pattern

    def forward(self, image):
        # need to check the type.
        self.scaling = cuda_like(self.scaling, image)
        return mosaick_multiply(self.scaling,
                                image,
                                self.mosaick_pattern)


class WhiteBalanceTemperature(nn.Module):

    def __init__(self,
                 new_temp,
                 new_tint=0.0,
                 orig_temp=6504,
                 orig_tint=0.0,
                 mosaick_pattern=None):
        """ WhiteBalancing with temperature parameterization.

        Args:
            new_temp: temperature to correct to. Can be scalar or 1D Tensor.
            new_tint: tint to correct to. Can be scalar or 1D Tensor.
            orig_temp: original temperature (default to D65)
            orig_tint: original tint (default to D65)
            mosaick_pattern: whether if the input has Bayer pattern.
        """
        super().__init__()
        # Make sure any scalars are converted to FloatTensor properly.
        self.new_temp = python_to_tensor(new_temp)
        self.new_tint = python_to_tensor(new_tint)
        self.orig_temp = python_to_tensor(orig_temp)
        self.orig_tint = python_to_tensor(orig_tint)
        self.mosaick_pattern = mosaick_pattern

    @staticmethod
    def _planckian_locus(T, tint):
        """Calculate Planckian Locus and its derivative in CIExyY.

        Args:
            T: Correlated Color Temp (in K) (Scalar or 1D tensor)
            tint: (to be implemented) (Scalar or 1D tensor)
        Returns:
            The white point in CIEXYZ space as a tensor of shape [batch x 3]
        """

        # formula from wikipedia
        def _blackbody_spectrum(l, T):
            """ Blackbody radiation spectrum

            See https://en.wikipedia.org/wiki/Planckian_locus.

            Args:
                l: wavelength in nanometer.
                T: temperature in Kelvin.
            """
            # See https://en.wikipedia.org/wiki/Planckian_locus.
            c2 = 1.4387773683E7
            l = l.unsqueeze(0)
            lT = l * T.unsqueeze(-1)
            return 1.0 / (torch.pow(l, 5) * (torch.exp(c2 / lT) - 1))

        def _diff_blackbody_spectrum(l, T):
            """ Temperature-derivative for blackbody spectrum function. This
                is used for tint where we find the perpendicular direction to
                the Planckian locus.
            """
            c2 = 1.4387773683E7
            l = l.unsqueeze(0)
            T = T.unsqueeze(-1)
            lT = l * T
            exp = torch.exp(c2 / (lT))
            return c2 * exp / (torch.pow(l, 6) * torch.pow(T * (exp - 1), 2))

        # Convert Scalar T into a 1D tensor
        if len(T.size()) < 1:
            T = T.unsqueeze(0)
        # Shape [batch x wavelength]
        M = _blackbody_spectrum(xyz_color_matching['lambda'], T)
        M_ = _diff_blackbody_spectrum(xyz_color_matching['lambda'], T)

        X = torch.sum(M.unsqueeze(1) * xyz_color_matching['xyz'].unsqueeze(0),
                      dim=-1)
        X_ = torch.sum(M_.unsqueeze(1) * xyz_color_matching['xyz'].unsqueeze(0),
                       dim=-1)
        Y = X[:, 1:2]
        Y_ = X_[:, 1:2]
        X_ = (X_ / Y) - (X / (Y * Y) * Y_)
        # switch X and Z so this is orthogonal
        X_[:, 0], X_[:, 2] = X_[:, 2], X_[:, 0]
        X_[:, 1] = 0
        X_ /= torch.sqrt(torch.sum(X_ ** 2, dim=1))
        # normalize Y to 1.
        X = X / X[:, 1:2] + tint.unsqueeze(-1) * X_
        return X

    def forward(self, image):
        X_orig = self._planckian_locus(self.orig_temp, self.orig_tint)
        X_new = self._planckian_locus(self.new_temp, self.new_tint)
        # The numerator is the original correction factor that makes D65
        # into [1, 1, 1] in sRGB. The XYZ2sRGB matrix encodes this, so as
        # a sanity check, XYZ2sRGB * X_D65 should equals 1.
        scaling = torch.matmul(XYZ2sRGB, X_new.t()) / \
                  torch.matmul(XYZ2sRGB, X_orig.t())
        # Transpose to [batch, 3]
        scaling = scaling.t()
        self._wb = WhiteBalance(scaling, self.mosaick_pattern)
        return self._wb(image)


class ColorSpaceConversionMatrix(nn.Module):

    def __init__(self, matrix):
        """ Linear color space conversion.

            Useful for converting between sRGB and YUV.

        Args:
            matrix: matrix to convert color space (should be 2-D Tensor).
                The conversion works as c_new = A * c_old, where c's are
                column vectors in each color space.
        """
        super().__init__()
        self.matrix = matrix

    def forward(self, image):
        self.matrix = cuda_like(self.matrix, image)
        return torch.einsum('ij,kjlm->kilm',
                            (self.matrix,
                             image)
                            )


class Saturation(nn.Module):

    def __init__(self, value):
        """ Adjust Saturation in YUV space

        Args:
            value: multiplier to the chrominance.
        """
        super().__init__()
        self.value = value
        self.rgb2yuv = ColorSpaceConversionMatrix(RGB2YUV)
        self.yuv2rgb = ColorSpaceConversionMatrix(YUV2RGB)

    def forward(self, image):
        image = self.rgb2yuv(image)
        image[:, 1:, ...] *= self.value
        image[:, 1:, ...] = torch.clamp(image[:, 1:, ...], -1.0, 1.0)
        image = self.yuv2rgb(image)
        return image


# Tone
class sRGBLikeGamma(nn.Module):
    def __init__(self, threshold, a, mult, gamma):
        """sRGB-like Gamma compression.

            Linear at low range then power gamma.
        Args:
            threshold: threshold under which the conversion becomes linear.
            a: constant factor to ensure continuity.
            mult: slope for the linear part.
            gamma: Gamma value.
        """
        super().__init__()
        self.threshold = threshold
        self.a = a
        self.mult = mult
        self.gamma = gamma

    def forward(self, image):
        mask = (image > self.threshold).float()
        image_lo = image * self.mult
        # 0.001 is to avoid funny thing at 0.
        image_hi = (1 + self.a) * torch.pow(image + 0.001, 1.0 / self.gamma) - self.a
        return mask * image_hi + (1 - mask) * image_lo


class UndosRGBLikeGamma(nn.Module):
    """ Linear at low range then power gamma.

        This is inverse of sRGBLikeGamma. See sRGBLikeGamma for detail.
    """

    def __init__(self, threshold, a, mult, gamma):
        super().__init__()
        self.threshold = threshold
        self.a = a
        self.mult = mult
        self.gamma = gamma

    def forward(self, image):
        mask = (image > self.threshold).float()
        image_lo = image / self.mult
        image_hi = torch.pow(image + self.a, self.gamma) / (1 + self.a)
        return mask * image_hi + (1 - mask) * image_lo


class sRGBGamma(sRGBLikeGamma):
    # See https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
    def __init__(self):
        super().__init__(threshold=0.0031308,
                         a=0.055,
                         mult=12.92,
                         gamma=2.4)


class UndosRGBGamma(UndosRGBLikeGamma):
    # See https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
    def __init__(self):
        super().__init__(threshold=0.04045,
                         a=0.055,
                         mult=12.92,
                         gamma=2.4)


class ProPhotoRGBGamma(sRGBLikeGamma):
    # See https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
    def __init__(self):
        super().__init__(threshold=1.0 / 512.0,
                         a=0.0,
                         mult=16.0,
                         gamma=1.8)


class UndoProPhotoRGBGamma(UndosRGBLikeGamma):
    # See https://en.wikipedia.org/wiki/SRGB#Specification_of_the_transformation
    def __init__(self):
        super().__init__(threshold=1.0 / 32.0,
                         a=0.0,
                         mult=16.0,
                         gamma=1.8)


class GammaCompression(nn.Module):

    def __init__(self, gamma):
        """ Pure power-law gamma compression.
        """
        super().__init__()
        gamma = python_to_tensor(gamma)
        self.gamma = expand_to_4d_batch(gamma)

    def forward(self, image):
        self.gamma = cuda_like(self.gamma, image)
        return (image + 0.0001).pow(self.gamma)


class UndoGammaCompression(nn.Module):

    def __init__(self, gamma):
        """ Inverse of GammaCompression.
        """
        super().__init__()
        gamma = python_to_tensor(gamma)
        self._gamma = GammaCompression(1.0 / gamma)

    def forward(self, image):
        return self._gamma(image)


class Gray18Gamma(nn.Module):

    def __init__(self, gamma):
        """ Applying gamma while keeping 18% gray constant.
        """
        super().__init__()
        gamma = python_to_tensor(gamma)
        self.gamma = expand_to_4d_batch(gamma)

    def forward(self, image):
        # mult x (0.18)^gamma = 0.18; 0.18 = 18% gray
        self.mult = FloatTensor([0.18]).pow(1.0 - self.gamma)
        self.gamma = cuda_like(self.gamma, image)
        self.mult = cuda_like(self.mult, image)
        return self.mult * torch.pow(image + 0.001, self.gamma)


class ToneCurve(nn.Module):

    def __init__(self, amount):
        """ Tone curve using cubic curve.

        The curve is assume to pass 0, 0.25-a, 0.5, 0.75+a, 1, where
        a is a parameter controlling the curve. For usability, the parameter 
        amount of 0 and 1 is mapped to a of 0 and 0.2.
        """
        super().__init__()
        self.amount = amount
        self.rgb2yuv = ColorSpaceConversionMatrix(RGB2YUV)
        self.yuv2rgb = ColorSpaceConversionMatrix(YUV2RGB)

    def forward(self, image):
        a = self.amount * 0.2
        self._A = -64.0 * a / 3.0
        self._B = 32.0 * a
        self._C = 1.0 - 32.0 * a / 3.0
        yuv = self.rgb2yuv(image)
        y = yuv[:, 0, ...]
        y_sqr = y * y
        y_cub = y_sqr * y
        y = self._A * y_cub + self._B * y_sqr + self._C * y
        yuv = yuv.clone()
        yuv[:, 0, ...] = y
        image = self.yuv2rgb(yuv)
        return image


class ToneCurveNZones(nn.Module):

    def __init__(self, ctrl_val):
        """ Tone curve using linear curve with N zone.

        Args:
            ctrl_val: list of values that specify control points. These
                are assumed to be equally spaced between 0 and 1.
        """
        super().__init__()
        self.ctrl_val = ctrl_val
        self.rgb2yuv = ColorSpaceConversionMatrix(RGB2YUV)
        self.yuv2rgb = ColorSpaceConversionMatrix(YUV2RGB)

    def forward(self, image):
        yuv = self.rgb2yuv(image)
        y = yuv[:, 0, ...]
        n_zones = len(self.ctrl_val) + 1
        val_scaling = 1.0 / n_zones
        in_val = torch.linspace(0, 1, n_zones + 1)
        out_val = [0] + [val_scaling * (i + 1 + self.ctrl_val[i]) for i in range(len(self.ctrl_val))] + [1]

        y_ = 0

        for i in range(len(in_val) - 1):
            # if statement for the boundary case, in case we have something negatives
            mask_lo = (y >= in_val[i]).float() if i > 0 else 1
            mask_hi = (y < in_val[i + 1]).float() if i < len(in_val) - 2 else 1
            mask = mask_lo * mask_hi
            slope = (out_val[i + 1] - out_val[i]) / (in_val[i + 1] - in_val[i])
            y_ += ((y - in_val[i]) * slope + out_val[i]) * mask

        yuv = yuv.clone()
        yuv[:, 0, ...] = y
        image = self.yuv2rgb(yuv)
        return image


class ToneCurveThreeZones(nn.Module):

    def __init__(self, highlight, midtone, shadow):
        """ Same as ToneCurveNZones but have different signature so that
            it is more explicit.
        """
        super().__init__()
        self.tc = ToneCurveNZones([shadow, midtone, highlight])

    def forward(self, image):
        return self.tc.forward(image)


class Quantize(nn.Module):

    def __init__(self, nbits=8):
        """ Quantize image to number of bits.
        """
        super().__init__()
        self.nbits = nbits

    def forward(self, image):
        self.mult = FloatTensor([2]).pow(self.nbits)
        self.mult = cuda_like(self.mult, image)
        return torch.floor(image * self.mult) / self.mult


class ExposureAdjustment(nn.Module):

    def __init__(self, nstops):
        """ Exposure adjustment by the stops.

        Args:
            nstops: number of stops to adjust exposure. Can be scalar or
                1D Tensor.
        """
        super().__init__()
        nstops = python_to_tensor(nstops)
        self.nstops = expand_to_4d_batch(nstops)

    def forward(self, image):
        self._multiplier = FloatTensor([2]).pow(self.nstops)
        self._multiplier = cuda_like(self._multiplier, image)
        return self._multiplier * image


class AffineExposure(nn.Module):

    def __init__(self, mult, add):
        """ Exposure adjustment with affine transform.

        This calculate exposure according to mult*L + add, where L is
        the current pixel value.

        Args:
            mult: Multiplier. Can be scalar or 1D Tensor.
            add: Additive constant. Can be scalar or 1D Tensor.
        """
        super().__init__()
        mult = python_to_tensor(mult)
        add = python_to_tensor(add)
        self._mult = expand_to_4d_batch(mult)
        self._add = expand_to_4d_batch(add)

    def forward(self, image):
        self._mult = cuda_like(self._mult, image)
        self._add = cuda_like(self._add, image)
        return self._mult * image + self._add


class AutoLevelNonDifferentiable(nn.Module):

    def __init__(self, blkpt=1, whtpt=99, max_mult=1.5):
        """ AutoLevel

            Non-differentiable because it uses percentile function.

            Args:
                blkpt: percentile used as black point.
                whtpt: percentile used as white point.
                max_mult: max multiplication factor to avoid over brightening
                    image.
        """
        super().__init__()
        self.blkpt = blkpt
        self.whtpt = whtpt
        self.max_mult = max_mult
        self.rgb2yuv = ColorSpaceConversionMatrix(RGB2YUV)
        self.yuv2rgb = ColorSpaceConversionMatrix(YUV2RGB)

    def forward(self, image):
        yuv = self.rgb2yuv(image)
        y = yuv[:, 0, ...].cpu().numpy()
        y = np.reshape(y, (y.shape[0], -1))
        blkpt = np.percentile(y, self.blkpt, axis=1)
        whtpt = np.percentile(y, self.whtpt, axis=1)

        mult = 1.0 / (whtpt - blkpt)
        if self.max_mult is not None:
            # if self.max_mult == "auto":
            # HACK: so that we can control both flow without additional switch.
            if self.max_mult < 0:
                mm = 4.0 * np.power(whtpt, -self.max_mult)
                mm = np.minimum(mm, 4.0)
                mm = np.maximum(mm, 1.0)
            else:
                mm = self.max_mult
            mult = np.minimum(mult, mm)
        mult = FloatTensor(mult).unsqueeze(-1).unsqueeze(-1)
        mult = cuda_like(mult, yuv)

        blkpt = FloatTensor(blkpt).unsqueeze(-1).unsqueeze(-1)
        blkpt = cuda_like(blkpt, yuv)

        # yuv[:, 0, ...] = (yuv[:, 0, ...] - blkpt) * mult
        image = (image - blkpt) * mult
        image = torch.clamp(image, 0.0, 1.0)
        return image


# Noises
class NoiseModule(nn.Module):
    """Base class for noise modules"""

    def get_noise_image(self, image):
        """ Return additive noise to the image.

        This function should return noise image with the standard deviation
        and the mosaick pattern baked in.
        """
        raise RuntimeError("This is a base class for noise modules. "
                           "Use one of its subclasses instead.")

    def forward(self, image):
        return image + self.get_noise_image(image)


class PoissonNoise(NoiseModule):

    def __init__(self, sigma, mosaick_pattern=None):
        """ Poisson noise

        Args:
            sigma: multiplier to the noise strength.
        """
        super().__init__()
        self.sigma = python_to_tensor(sigma)
        self.mosaick_pattern = mosaick_pattern

    def get_noise_image(self, image):
        noise_image = torch.randn_like(image)
        noise_image *= torch.sqrt(torch.clamp(image, min=0.0))
        self.sigma = cuda_like(self.sigma, image)
        return mosaick_multiply(self.sigma, noise_image, self.mosaick_pattern)


class GaussianNoise(NoiseModule):

    def __init__(self, sigma, mosaick_pattern=None):
        """ Gaussian noise

        Args:
            sigma: noise STD.
        """
        super().__init__()
        self.sigma = python_to_tensor(sigma)
        self.mosaick_pattern = mosaick_pattern

    def get_noise_image(self, image):
        noise_image = torch.randn_like(image)
        self.sigma = cuda_like(self.sigma, image)
        return mosaick_multiply(self.sigma, noise_image, self.mosaick_pattern)


class GaussPoissonMixtureNoise(NoiseModule):

    def __init__(self, sigma_p, sigma_g, mosaick_pattern=None):
        """ Gaussian and poisson noise mixture.

        Args:
            sigma_p: poisson noise multiplication..
            sigma_g: noise gaussian STD.
        """
        super().__init__()
        self.mosaick_pattern = mosaick_pattern
        self.sigma_p = sigma_p
        self.sigma_g = sigma_g
        self._poisson = PoissonNoise(self.sigma_p, self.mosaick_pattern)
        self._gauss = GaussianNoise(self.sigma_g, self.mosaick_pattern)

    def get_noise_image(self, image):
        return self._poisson.get_noise_image(image) + \
               self._gauss.get_noise_image(image)


# Other artifacts.
class JPEGCompression(nn.Module):
    DCT_BLOCK_SIZE = 8

    # TODO: Support batch for different quality.
    def __init__(self, quality):
        """ JPEGCompression with integer quality.

        Args:
            quality: integer between 0 and 12 (highest quality).
                This selects quantization table to use. See constant.py
                for detail.
        """
        # Quality must be integer between 0 and 12.
        super().__init__()
        quality = int(quality)
        # Add batch and channel dimension
        self.DCT_coeff_block = DCT_coeff.clone().unsqueeze(0).unsqueeze(0)

        self.quantization_lum = photoshop_jpeg_quantization_lum[quality]
        self.quantization_chrom = photoshop_jpeg_quantization_chrom[quality]

        self.quantization_lum = self.quantization_lum \
            .unsqueeze(0).unsqueeze(0) \
            .unsqueeze(-1).unsqueeze(-1)
        self.quantization_chrom = self.quantization_chrom \
            .unsqueeze(0).unsqueeze(0) \
            .unsqueeze(-1).unsqueeze(-1)
        self.downsample_chrom = photoshop_chroma_subsampling[quality]

        self.rgb2yuv = ColorSpaceConversionMatrix(RGB2YUV)
        self.yuv2rgb = ColorSpaceConversionMatrix(YUV2RGB)

    @staticmethod
    def _tile_sum(arr, dct_block):
        """Do the cumulative sum in tiles over the last two dimensions.

        input should be shaped (batch, ch, blk_sz, blk_sz, im_h, im_w)
        output will be (batch, ch, blk_sz, blk_sz, n_blk_h, n_blk_w)
        """
        verbose = False
        dct_block_size = dct_block.size(-1)
        # allocating a temp array seems helpful, maybe because it doesn't
        # have to write to the original array, which would result in more
        # cache misses.
        res = torch.zeros((arr.size(0),
                           arr.size(1),
                           dct_block_size, dct_block_size,
                           int(arr.size(4) / dct_block_size),
                           arr.size(5)))
        # also multiply DCT coefficient here because actually repeating
        # in two dim and multiply is very slow.
        dct_block = dct_block.repeat(1, 1, 1, 1, 1, int(arr.size(5) / dct_block_size))
        # Sum in height and multiply.
        for i in range(dct_block_size):
            res += arr[..., i::dct_block_size, :] * dct_block[..., i:(i + 1), :]
        # Sum in width
        for i in range(dct_block_size - 1):
            res[..., :, (i + 1)::dct_block_size] += res[..., :, i::dct_block_size]
        # Slice the array
        # now DCT should have dimension (batch, ch, 8, 8, n_blk_h, n_blk_w)
        res = res[..., :, (dct_block_size - 1)::dct_block_size]
        return res

    @staticmethod
    def _tile_to_image(arr):
        """Takes arr of shape (batch, ch, blk_sz, blk_sz, n_blk_h, n_blk_w),
        and reshape it so that it is (batch, ch, im_h, im_w)
        """
        # For readability
        dct_block_size = JPEGCompression.DCT_BLOCK_SIZE
        n_blk_h = int(arr.size(-2))
        n_blk_w = int(arr.size(-1))
        # reshape it, assume reshape does it in C-order, last element changing fastest.
        # Rearrange it so that it is
        # (batch, ch, n_blk_h, v, n_blk_w, u)
        arr = arr.permute(0, 1, 4, 2, 5, 3)
        # dct is now (batch, ch, y, x, v, u)
        arr = arr.contiguous()
        arr = arr.view(arr.size(0),
                       arr.size(1),
                       n_blk_h * dct_block_size,
                       n_blk_w * dct_block_size)
        return arr

    def _compress(self, image, quantization_matrix):
        # convert to -128 - 127 range
        image = (image * 255.0) - 128.0
        # For readability
        dct_block_size = JPEGCompression.DCT_BLOCK_SIZE
        # pad image
        im_h = int(image.size(-2))
        im_w = int(image.size(-1))
        n_blk_h = int(np.ceil(im_h / dct_block_size))
        n_blk_w = int(np.ceil(im_w / dct_block_size))
        n_pad_h = n_blk_h * dct_block_size - image.size(-2)
        n_pad_w = n_blk_w * dct_block_size - image.size(-1)
        # pad image
        image = torch.nn.functional.pad(image, (0, n_pad_w, 0, n_pad_h))
        # Add u, v dimension
        image = image.unsqueeze(-3).unsqueeze(-3)
        # Compute DCT
        # Sum within each tile.
        dct = self._tile_sum(image, self.DCT_coeff_block)
        # Quantize
        dct = torch.round(dct / quantization_matrix) * quantization_matrix
        # reshape it so that this becomes a u-v image.
        dct = self._tile_to_image(dct).unsqueeze(-3).unsqueeze(-3)
        # DCT should be (batch, ch, 8, 8, im_h, im_w)
        # do the sum in u, v
        dct = self._tile_sum(dct, self.DCT_coeff_block.permute(0, 1, 4, 5, 2, 3))
        dct = self._tile_to_image(dct)
        # Undo padding.
        dct = dct[..., :im_h, :im_w]
        # convert back to 0-1 range
        dct = (dct + 128.0) / 255.0
        return dct

    def forward(self, image):
        self.quantization_lum = cuda_like(self.quantization_lum, image)
        self.DCT_coeff_block = cuda_like(self.DCT_coeff_block, image)
        image_yuv = self.rgb2yuv(image)
        image_y = image_yuv[:, 0:1, ...]
        image_uv = image_yuv[:, 1:, ...]
        # Compress luminance.
        image_y = self._compress(image_y, self.quantization_lum)
        # Compress the chrominance.
        if self.downsample_chrom:
            uv_size = image_uv.size()
            image_uv = nn.functional.interpolate(image_uv, scale_factor=0.5)
        image_uv = self._compress(image_uv, self.quantization_chrom)
        if self.downsample_chrom:
            image_uv = nn.functional.interpolate(image_uv, size=uv_size[-2:])
        image_yuv = torch.cat((image_y, image_uv), dim=1)
        image = self.yuv2rgb(image_yuv)
        return image


class ChromaticAberration(nn.Module):

    def __init__(self, scaling):
        """Chromatic Aberration

        Args:
            scaling: This class scales R and B channel with factor of scaling and 1/scaling
                respectively.
        """
        super().__init__()
        self.scaling = expand_to_4d_batch(python_to_tensor(scaling))

    @staticmethod
    def _scale(image, scaling):
        # create the affine matrix.
        theta = torch.zeros((image.size(0), 2, 3))
        # diagonal entry
        theta[:, 0, 0] = scaling
        theta[:, 1, 1] = scaling
        theta = cuda_like(theta, image)
        grid = nn.functional.affine_grid(theta, image.size())
        return nn.functional.grid_sample(image, grid, padding_mode="border")

    def forward(self, image):
        # R
        output_img = image.clone()
        output_img[:, 0:1, ...] = self._scale(image[:, 0:1, ...],
                                              self.scaling)
        # B
        output_img[:, 2:3, ...] = self._scale(image[:, 2:3, ...],
                                              1.0 / self.scaling)
        return output_img


class PixelClip(nn.Module):
    """ Module for clipping pixel value.
    """

    def forward(self, image):
        return torch.clamp(image, 0.0, 1.0)


class RepairHotDeadPixel(nn.Module):

    # Adapt from https://github.com/letmaik/rawpy/blob/291afa870727f759a7bb68d756e4603806a466a4/rawpy/enhance.py
    def __init__(self, threshold=0.2, median_class="MedianFilterNonDifferentiable"):
        """ Repair hot pixel with median filter.

        Args:
            threshold: Difference to be considered as hot/dead pixels.
        """
        super().__init__()
        median_classes = {"MedianFilterNonDifferentiable": MedianFilterNonDifferentiable,
                          }
        self.median = median_classes[median_class](3)
        self.threshold = threshold

    def _repair_one_channel(self, rawslice):
        med = self.median(rawslice.clone())
        # detect possible bad pixels
        candidates = torch.abs(rawslice - med) > self.threshold
        candidates = candidates.float()
        candidates = cuda_like(candidates, rawslice)
        return (1.0 - candidates) * rawslice + candidates * med

    def forward(self, image):
        # we have bayer
        if image.size(1) == 1:
            # we have 4 colors (two greens are always seen as two colors)
            for offset_y in [0, 1]:
                for offset_x in [0, 1]:
                    rawslice = image[..., offset_y::2, offset_x::2]
                    rawslice = self._repair_one_channel(rawslice)
                    image[..., offset_y::2, offset_x::2] = rawslice
        else:
            # do it per channel
            for i in range(image.size(1)):
                rawslice = image[:, i:(i + 1), ...]
                rawslice = self._repair_one_channel(rawslice)
                image[:, i:(i + 1), ...] = rawslice
        return image


class PerChannelBlur(nn.Module):

    def __init__(self, kern):
        """ Blur applied to each channel individually.

        Args:
            kern: 2D tensors representing the blur kernel.
        """
        super().__init__()
        self.kern = kern

    def forward(self, image):
        self.kern = FloatTensor(self.kern).unsqueeze(0).unsqueeze(0)
        self.kern = cuda_like(self.kern, image)
        n_channel = image.size(1)
        padding = []
        for i in range(2):
            # See https://stackoverflow.com/questions/51131821/even-sized-kernels-with-same-padding-in-tensorflow
            sz = self.kern.size(-1 - i)
            total_pad = int(sz - 1)
            p0 = int(total_pad / 2)
            p1 = total_pad - p0
            padding += [p0, p1]
        # Manually pad.
        image = nn.functional.pad(image, padding, mode='reflect')
        return nn.functional.conv2d(image,
                                    self.kern.expand(n_channel,
                                                     -1, -1, -1),
                                    groups=n_channel)


class SeparablePerChannelBlur(nn.Module):

    def __init__(self, kern_x, kern_y=None):
        """Same as PerChannelBlur, but separable kernel.

        This is much faster. Useful for when we have separable kernel such as
        Gaussian.

        Args:
            kern_x: 1D tensor representing x-direction kernel.
            kern_y: 1D tensor representing y-direction kernel. If None, use
                the same thing as kern_x.
        """
        super().__init__()
        if kern_y is None:
            kern_y = kern_x
        self.kern_x = kern_x
        self.kern_y = kern_y

    def forward(self, image):
        self.kern_x = FloatTensor(self.kern_x).unsqueeze(0).unsqueeze(0)
        self.kern_y = FloatTensor(self.kern_y).unsqueeze(0).unsqueeze(0)
        self.kern_x = cuda_like(self.kern_x, image)
        self.kern_y = cuda_like(self.kern_y, image)
        n_channel = image.size(1)
        padding = []
        kern_sz = (self.kern_y.size(-1), self.kern_x.size(-1))
        for i in range(len(kern_sz)):
            # See https://stackoverflow.com/questions/51131821/even-sized-kernels-with-same-padding-in-tensorflow
            sz = kern_sz[-1 - i]
            total_pad = int(sz - 1)
            p0 = int(total_pad / 2)
            p1 = total_pad - p0
            padding += [p0, p1]
        # Manually pad.
        image_sz = image.size()
        image = nn.functional.pad(image, padding, mode='reflect')
        image = image.contiguous().view(-1,
                                        image.size(-2),
                                        image.size(-1))
        # Do convolution in each direction
        # width, b, height
        image = image.permute(2, 0, 1)
        image = nn.functional.conv1d(image,
                                     self.kern_y.expand(image.size(1),
                                                        -1, -1),
                                     groups=image.size(1))
        # height, b, width
        image = image.permute(2, 1, 0)
        image = nn.functional.conv1d(image,
                                     self.kern_x.expand(image.size(1),
                                                        -1, -1),
                                     groups=image.size(1))
        # b, height, width
        image = image.permute(1, 0, 2)
        return image.view(image_sz)


class MotionBlur(PerChannelBlur):

    # TODO: Think about how to generate big blur without a giant kernel.
    # Seems like this might not be possible
    def __init__(self, amt, direction,
                 kernel_sz=10,
                 dynrange_th=None,
                 dynrange_boost=100
                 ):
        """Motion Blur

        Args:
            amt: (list or number) list of amount of motion blur in pixel.
            direction: (list or number) direction of motion in degrees.
            kernel_sz: max size of kernel for performance consideration.
            dynrange_th: threshold above which will get boosted to simulate
                overexposed pixels. (See Burst Image Deblurring Using
                Permutation Invariant Convolutional Neural Networks by Aittala
                et al. 2018).
            dynrange_boost: Multiplicative factor used to boost dynamic range.
        """
        # normalize input into a good format.
        amt = number_to_list(amt)
        direction = number_to_list(direction)
        assert len(amt) == len(direction)
        # Create the blur kernel.
        origin = np.array([0.0, 0.0]).astype('float')
        pts = [origin]
        min_x = max_x = min_y = max_y = 0.0
        for idx in range(len(amt)):
            d = direction[idx] * np.pi / 180.0
            vec = np.array((np.cos(d), np.sin(d))) * amt[idx]
            pt = pts[-1] + vec
            x, y = pt[0], pt[1]
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
            pts.append(pt)
        cv_bit_shift = 8
        mult = np.power(2, cv_bit_shift)
        if kernel_sz is None:
            # figure out kernel_sz
            ksz_x = max(max_x - min_x + 2, 8)
            ksz_y = max(max_y - min_y + 2, 8)
        else:
            ksz_x = ksz_y = kernel_sz

        ksz_x = int(ksz_x)
        ksz_y = int(ksz_y)
        kern = np.zeros((ksz_y, ksz_x)).astype('uint8')
        pts = np.array(pts)
        pts[:, 0] -= min_x
        pts[:, 1] -= min_y
        pts *= mult
        # TODO: Remove cv2 dependencies and use skimage instead.
        # LINE_AA only works with uint8 kernel, but there is a bug that it
        # only draws the first segment in this mode
        cv2.polylines(kern, np.int32([pts]), isClosed=False,
                      color=1.0, lineType=8,
                      thickness=1, shift=cv_bit_shift)
        kern = kern.astype('float32')
        kern = kern / kern.sum()
        super().__init__(kern)
        self.dynrange_th = dynrange_th
        self.dynrange_boost = dynrange_boost
        if dynrange_th is not None:
            self.rgb2yuv = ColorSpaceConversionMatrix(RGB2YUV)

    def forward(self, image):
        if self.dynrange_th is not None:
            y = self.rgb2yuv(image)[:, 0:1, ...]
            mask = y > self.dynrange_th
            mask = cuda_like(mask.float(), image)
            image = image * (1.0 + mask * self.dynrange_boost)
        image = super().forward(image)
        if self.dynrange_th is not None:
            image = torch.clamp(image, 0.0, 1.0)
        return image


class GaussianBlur(SeparablePerChannelBlur):

    def __init__(self, sigma_x, sigma_y=None,
                 sz_x=None, sz_y=None):
        """Channel-wise Gaussian Blur.

        Args:
            sigma_x: stdev in x-direction.
            sigma_y: stdev in y-direction. (default: sigma_x)
            sz_x = kernel size in x (default: twice sigma_x)
            sz_y = kernel size in y (default: twice sigma_y)
        """
        if sigma_y is None:
            sigma_y = sigma_x
        if sz_x is None:
            sz_x = max(int(2.0 * sigma_x), 1)
        if sz_y is None:
            sz_y = max(int(2.0 * sigma_y), 1)
        super().__init__(None, None)
        self.sz_x = sz_x
        self.sz_y = sz_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def forward(self, image):
        self.kern_x = gausskern1d(self.sigma_x, self.sz_x)
        self.kern_y = gausskern1d(self.sigma_y, self.sz_y)
        return super().forward(image)


class Rotation90Mult(nn.Module):

    def __init__(self, angle):
        """ Rotate image in multiples of 90.
        """
        super().__init__()
        self.angle = int(angle) % 360
        if self.angle not in [0, 90, 180, 270]:
            raise ValueError("Angle must be multiple of 90 degrees")

    def forward(self, image):
        if self.angle == 0:
            return image
        elif self.angle == 90:
            return image.transpose(2, 3).flip(2)
        elif self.angle == 270:
            return image.transpose(2, 3).flip(3)
        elif self.angle == 180:
            return image.flip(2).flip(3)
        else:
            raise ValueError("Angle must be multiple of 90 degrees")
