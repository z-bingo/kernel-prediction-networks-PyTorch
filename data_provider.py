import glob
import inspect
import os
import zlib
from time import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from torch import FloatTensor

from data_generation.pipeline import ImageDegradationPipeline
from utils.image_utils import bayer_crop_tensor
from utils.training_util import read_config

DEBUG_TIME = False


def _configspec_path():
    current_dir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    return os.path.join(current_dir,
                        'dataset_specs/data_configspec.conf')


class OnTheFlyDataset(data.Dataset):
    def __init__(self,
                 config_file,
                 config_spec=None,
                 blind=False,
                 cropping="random",
                 cache_dir=None,
                 use_cache=False,
                 dataset_name="synthetic"):
        """ Dataset for generating degraded images on the fly.

        Args:
            pipeline_configs: dictionary of boolean flags controlling how
                pipelines are created.
            pipeline_param_ranges: dictionary of ranges of params.
            patch_dir: directory to load linear patches.
            config_file: path to data config file
            im_size: tuple of (w, h)
            config_spec: path to data config spec file
            cropping: cropping mode ["random", "center"]
        """
        super().__init__()
        if config_spec is None:
            config_spec = _configspec_path()
        config = read_config(config_file, config_spec)
        # self.config_file = config_file
        # dictionary of dataset configs
        self.dataset_configs = config['dataset_configs']
        # directory to load linear patches
        patch_dir = self.dataset_configs['dataset_dir']
        # dictionary of boolean flags controlling how pipelines are created
        # (see data_configspec for detail).
        self.pipeline_configs = config['pipeline_configs']
        # dictionary of ranges of params (see data_configspec for detail).
        self.pipeline_param_ranges = config['pipeline_param_ranges']

        file_list = glob.glob(os.path.join(patch_dir,
                                           '*.pth'))
        file_list = [os.path.basename(f) for f in file_list]
        file_list = [os.path.splitext(f)[0] for f in file_list]
        self.file_list = sorted(file_list, key=lambda x: zlib.adler32(x.encode('utf-8')))
        # print(self.file_list)
        # self.pipeline_param_ranges = pipeline_param_ranges
        # self.pipeline_configs = pipeline_configs
        # print('Data Pipeline Configs: ', self.pipeline_configs)
        # print('Data Pipeline Param Ranges: ', self.pipeline_param_ranges)
        # some variables about the setting of dataset
        self.data_root = patch_dir
        self.im_size = self.dataset_configs['patch_size']  # the size after down-sample
        extra_for_bayer = 2  # extra size used for the random choice for bayer pattern
        self.big_jitter = self.dataset_configs['big_jitter']
        self.small_jitter = self.dataset_configs['small_jitter']
        self.down_sample = self.dataset_configs['down_sample']
        # image size corresponding to original image (include big jitter)
        self.im_size_upscale = (self.im_size + 2 * self.big_jitter + extra_for_bayer) * self.down_sample
        # from big jitter image to real image with extra pixels to random choose the bayer pattern
        self.big_restore_upscale = self.big_jitter * self.down_sample
        # the shift pixels of small jitter within upscale
        self.small_restore_upscale = self.small_jitter * self.down_sample
        # from big jitter images to small jitter images
        self.big2small_upscale = (self.big_jitter - self.small_jitter) * self.down_sample
        #
        self.im_size_extra = (self.im_size + extra_for_bayer) * self.down_sample
        # blind estimate?
        self.blind = blind
        # others
        self.cropping = cropping
        self.use_cache = use_cache
        self.cache_dir = cache_dir

        sz = "{}x{}".format(self.im_size, self.im_size) \
            if self.im_size is not None else "None"
        self.dataset_name = "_".join([dataset_name, sz])

        # add the codes by Bin Zhang
        self.burst_length = self.dataset_configs['burst_length']

    def _get_filename(self, idx):
        # folder = os.path.join(self.cache_dir, self.dataset_name)
        folder = self.cache_dir
        if not os.path.exists(folder):
            os.makedirs(folder)
        # filename = os.path.join(folder, self.dataset_name + "_{:06d}.pth".format(idx))
        filename = os.path.join(folder, "{:06d}.pth".format(idx))
        return filename

    def _save_tensor(self, tensor_dicts, idx):
        filename = self._get_filename(idx)
        try:
            torch.save(tensor_dicts, filename)
        except OSError as e:
            print("Warning write failed.")
            print(e)

    def _load_tensor(self, idx):
        filename = self._get_filename(idx)
        return torch.load(filename)

    def _random_log_uniform(self, a, b):
        if self.legacy_uniform:
            return np.random.uniform(a, b)
        val = np.random.uniform(np.log(a), np.log(b))
        return np.exp(val)

    def _randomize_parameter(self):
        if "use_log_uniform" in self.pipeline_configs:
            self.legacy_uniform = not self.pipeline_configs["use_log_uniform"]
        else:
            self.legacy_uniform = True

        exp_adjustment = np.random.uniform(self.pipeline_param_ranges["min_exposure_adjustment"],
                                           self.pipeline_param_ranges["max_exposure_adjustment"])
        poisson_k = self._random_log_uniform(self.pipeline_param_ranges["min_poisson_noise"],
                                             self.pipeline_param_ranges["max_poisson_noise"])
        read_noise_sigma = self._random_log_uniform(self.pipeline_param_ranges["min_gaussian_noise"],
                                                    self.pipeline_param_ranges["max_gaussian_noise"])
        chromatic_aberration = np.random.uniform(self.pipeline_param_ranges["min_chromatic_aberration"],
                                                 self.pipeline_param_ranges["max_chromatic_aberration"])
        motionblur_segment = np.random.randint(self.pipeline_param_ranges["min_motionblur_segment"],
                                               self.pipeline_param_ranges["max_motionblur_segment"])
        motion_blur = []
        motion_blur_dir = []
        for i in range(motionblur_segment):
            motion_blur.append(np.random.uniform(self.pipeline_param_ranges["min_motion_blur"],
                                                 self.pipeline_param_ranges["max_motion_blur"])
                               )
            motion_blur_dir.append(np.random.uniform(0.0, 360.0))
        jpeg_quality = np.random.randint(self.pipeline_param_ranges["min_jpeg_quality"],
                                         self.pipeline_param_ranges["max_jpeg_quality"])
        denoise_sigma_s = self._random_log_uniform(self.pipeline_param_ranges["min_denoise_sigma_s"],
                                                   self.pipeline_param_ranges["max_denoise_sigma_s"])
        denoise_sigma_r = self._random_log_uniform(self.pipeline_param_ranges["min_denoise_sigma_r"],
                                                   self.pipeline_param_ranges["max_denoise_sigma_r"])
        denoise_color_sigma_ratio = self._random_log_uniform(
            self.pipeline_param_ranges["min_denoise_color_sigma_ratio"],
            self.pipeline_param_ranges["max_denoise_color_sigma_ratio"])
        denoise_color_range_ratio = self._random_log_uniform(
            self.pipeline_param_ranges["min_denoise_color_range_ratio"],
            self.pipeline_param_ranges["max_denoise_color_range_ratio"])
        unsharp_amount = np.random.uniform(self.pipeline_param_ranges["min_unsharp_amount"],
                                           self.pipeline_param_ranges["max_unsharp_amount"])
        denoise_median_sz = np.random.randint(self.pipeline_param_ranges["min_denoise_median_sz"],
                                              self.pipeline_param_ranges["max_denoise_median_sz"])
        quantize_bits = np.random.randint(self.pipeline_param_ranges["min_quantize_bits"],
                                          self.pipeline_param_ranges["max_quantize_bits"])
        wavelet_sigma = np.random.uniform(self.pipeline_param_ranges["min_wavelet_sigma"],
                                          self.pipeline_param_ranges["max_wavelet_sigma"])
        motionblur_th = np.random.uniform(self.pipeline_param_ranges["min_motionblur_th"],
                                          self.pipeline_param_ranges["max_motionblur_th"])
        motionblur_boost = self._random_log_uniform(self.pipeline_param_ranges["min_motionblur_boost"],
                                                    self.pipeline_param_ranges["max_motionblur_boost"])
        return dict(
            exp_adjustment=exp_adjustment,
            poisson_k=poisson_k,
            read_noise_sigma=read_noise_sigma,
            chromatic_aberration=chromatic_aberration,
            motion_blur=motion_blur,
            motion_blur_dir=motion_blur_dir,
            jpeg_quality=jpeg_quality,
            denoise_sigma_s=denoise_sigma_s,
            denoise_sigma_r=denoise_sigma_r,
            denoise_color_sigma_ratio=denoise_color_sigma_ratio,
            denoise_color_range_ratio=denoise_color_range_ratio,
            unsharp_amount=unsharp_amount,
            denoise_median=denoise_median_sz,
            quantize_bits=quantize_bits,
            wavelet_sigma=wavelet_sigma,
            motionblur_th=motionblur_th,
            motionblur_boost=motionblur_boost,
        )

    @staticmethod
    def _create_pipeline(exp_adjustment,
                         poisson_k,
                         read_noise_sigma,
                         chromatic_aberration,
                         motion_blur_dir,
                         jpeg_quality,
                         denoise_sigma_s,
                         denoise_sigma_r,
                         denoise_color_sigma_ratio,
                         unsharp_amount,
                         denoise_color_only,
                         demosaick,
                         denoise,
                         jpeg_compression,
                         use_motion_blur,
                         use_chromatic_aberration,
                         use_unsharp_mask,
                         exposure_correction,
                         quantize,
                         quantize_bits=8,
                         denoise_guide_transform=None,
                         denoise_n_iter=1,
                         demosaick_use_median=False,
                         demosaick_n_iter=0,
                         use_median_denoise=False,
                         median_before_bilateral=False,
                         denoise_median=None,
                         denoise_median_ratio=1.0,
                         denoise_median_n_iter=1,
                         demosaicked_input=True,
                         log_blackpts=0.004,
                         bilateral_class="DenoisingSKImageBilateralNonDifferentiable",
                         demosaick_class="AHDDemosaickingNonDifferentiable",
                         demosaick_ahd_delta=2.0,
                         demosaick_ahd_sobel_sz=3,
                         demosaick_ahd_avg_sz=3,
                         use_wavelet=False,
                         wavelet_family="db2",
                         wavelet_sigma=None,
                         wavelet_th_method="BayesShrink",
                         wavelet_levels=None,
                         motion_blur=None,
                         motionblur_th=None,
                         motionblur_boost=None,
                         motionblur_segment=1,
                         debug=False,
                         bayer_crop_phase=None,
                         saturation=None,
                         use_autolevel=False,
                         autolevel_max=1.5,
                         autolevel_blk=1,
                         autolevel_wht=99,
                         denoise_color_range_ratio=1,
                         wavelet_last=False,
                         wavelet_threshold=None,
                         wavelet_filter_chrom=True,
                         post_tonemap_class=None,
                         post_tonemap_amount=None,
                         pre_tonemap_class=None,
                         pre_tonemap_amount=None,
                         post_tonemap_class2=None,
                         post_tonemap_amount2=None,
                         repair_hotdead_pixel=False,
                         hot_px_th=0.2,
                         white_balance=False,
                         white_balance_temp=6504,
                         white_balance_tint=0,
                         use_tone_curve3zones=False,
                         tone_curve_highlight=0.0,
                         tone_curve_midtone=0.0,
                         tone_curve_shadow=0.0,
                         tone_curve_midshadow=None,
                         tone_curve_midhighlight=None,
                         unsharp_radius=4.0,
                         unsharp_threshold=3.0,
                         **kwargs):

        # Define image degradation pipeline
        # add motion blur and chromatic aberration
        configs_degrade = []
        # Random threshold
        if demosaicked_input:
            # These are features that only make sense to simulate in 
            # demosaicked input.
            if use_motion_blur:
                configs_degrade += [
                    ('MotionBlur', {'amt': motion_blur,
                                    'direction': motion_blur_dir,
                                    'kernel_sz': None,
                                    'dynrange_th': motionblur_th,
                                    'dynrange_boost': motionblur_boost,
                                    }
                     )
                ]
            if use_chromatic_aberration:
                configs_degrade += [
                    ('ChromaticAberration', {'scaling': chromatic_aberration}),
                ]

        configs_degrade.append(('ExposureAdjustment', {'nstops': exp_adjustment}))
        if demosaicked_input:
            if demosaick:
                configs_degrade += [
                    ('BayerMosaicking', {}),
                ]
                mosaick_pattern = 'bayer'
            else:
                mosaick_pattern = None
        else:
            mosaick_pattern = 'bayer'

        # Add artificial noise.
        configs_degrade += [
            ('PoissonNoise', {'sigma': poisson_k, 'mosaick_pattern': mosaick_pattern}),
            ('GaussianNoise', {'sigma': read_noise_sigma, 'mosaick_pattern': mosaick_pattern}),
        ]

        if quantize:
            configs_degrade += [
                ('PixelClip', {}),
                ('Quantize', {'nbits': quantize_bits}),
            ]
        if repair_hotdead_pixel:
            configs_degrade += [
                ("RepairHotDeadPixel", {"threshold": hot_px_th}),
            ]

        if demosaick:
            configs_degrade += [
                (demosaick_class, {'use_median_filter': demosaick_use_median,
                                   'n_iter': demosaick_n_iter,
                                   'delta': demosaick_ahd_delta,
                                   'sobel_sz': demosaick_ahd_sobel_sz,
                                   'avg_sz': demosaick_ahd_avg_sz,
                                   }),
                ('PixelClip', {}),
            ]
        if white_balance:
            configs_degrade += [
                ('WhiteBalanceTemperature', {"new_temp": white_balance_temp,
                                             "new_tint": white_balance_tint,
                                             }),
            ]
        if pre_tonemap_class is not None:
            kw = "gamma" if "Gamma" in pre_tonemap_class else "amount"
            configs_degrade += [
                (pre_tonemap_class, {kw: pre_tonemap_amount})
            ]
        if use_autolevel:
            configs_degrade.append(('AutoLevelNonDifferentiable', {'max_mult': autolevel_max,
                                                                   'blkpt': autolevel_blk,
                                                                   'whtpt': autolevel_wht,
                                                                   }))
        denoise_list = []
        if denoise:
            denoise_list.append([
                ('PixelClip', {}),
                (bilateral_class, {'sigma_s': denoise_sigma_s,
                                   'sigma_r': denoise_sigma_r,
                                   'color_sigma_ratio': denoise_color_sigma_ratio,
                                   'color_range_ratio': denoise_color_range_ratio,
                                   'filter_lum': not denoise_color_only,
                                   'n_iter': denoise_n_iter,
                                   'guide_transform': denoise_guide_transform,
                                   '_bp': log_blackpts,
                                   }),
                ('PixelClip', {}),
            ])

        if use_median_denoise:
            # TODO: Fix this.
            # Special value because our config can't specify list of list
            if denoise_median == -1:
                denoise_median = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
            if debug:
                print("Denoising with Median Filter")
            denoise_list.append([
                ('DenoisingMedianNonDifferentiable', {'neighbor_sz': denoise_median,
                                                      'color_sigma_ratio': denoise_median_ratio,
                                                      'n_iter': denoise_median_n_iter,
                                                      }),
            ])

        if median_before_bilateral:
            denoise_list = denoise_list[::-1]
        if use_wavelet:
            # always do wavelet first.
            wavelet_config = [
                ('PixelClip', {}),
                ("DenoisingWaveletNonDifferentiable", {'sigma_s': wavelet_th_method,
                                                       'sigma_r': wavelet_sigma,
                                                       'color_sigma_ratio': wavelet_family,
                                                       'filter_lum': True,
                                                       'n_iter': wavelet_levels,
                                                       'guide_transform': denoise_guide_transform,
                                                       '_bp': wavelet_threshold,
                                                       'filter_chrom': wavelet_filter_chrom,
                                                       }),
                ('PixelClip', {}),
            ]
            if wavelet_last:
                denoise_list.append(wavelet_config)
            else:
                denoise_list.insert(0, wavelet_config)

        for i in range(len(denoise_list)):
            configs_degrade += denoise_list[i]
        if post_tonemap_class is not None:
            kw = "gamma" if "Gamma" in post_tonemap_class else "amount"
            configs_degrade += [
                (post_tonemap_class, {kw: post_tonemap_amount})
            ]
        if post_tonemap_class2 is not None:
            kw = "gamma" if "Gamma" in post_tonemap_class2 else "amount"
            configs_degrade += [
                (post_tonemap_class2, {kw: post_tonemap_amount2})
            ]
        if use_tone_curve3zones:
            ctrl_val = [t for t in [tone_curve_shadow,
                                    tone_curve_midshadow,
                                    tone_curve_midtone,
                                    tone_curve_midhighlight,
                                    tone_curve_highlight] if t is not None]
            configs_degrade += [
                ('ToneCurveNZones', {'ctrl_val': ctrl_val,
                                     }),
                ('PixelClip', {}),
            ]

        if use_unsharp_mask:
            configs_degrade += [
                ('Unsharpen', {'amount': unsharp_amount,
                               'radius': unsharp_radius,
                               'threshold': unsharp_threshold}),
                ('PixelClip', {}),
            ]

        if saturation is not None:
            configs_degrade.append(('Saturation', {'value': saturation}))

        # things that happens after camera apply denoising, etc. 
        if jpeg_compression:
            configs_degrade += [
                ('sRGBGamma', {}),
                ('Quantize', {'nbits': 8}),
                ('PixelClip', {}),
                ('JPEGCompression', {"quality": jpeg_quality}),
                ('PixelClip', {}),
                ('UndosRGBGamma', {}),
                ('PixelClip', {}),
            ]
        else:
            if quantize:
                configs_degrade += [
                    ('Quantize', {'nbits': 8}),
                    ('PixelClip', {}),
                ]

        if exposure_correction:
            # Finally do exposure correction of weird jpeg-compressed image to get crappy images.
            configs_degrade.append(('ExposureAdjustment', {'nstops': -exp_adjustment}))
            target_pipeline = None
        else:
            configs_target = [
                ('ExposureAdjustment', {'nstops': exp_adjustment}),
                ('PixelClip', {}),
            ]
            target_pipeline = ImageDegradationPipeline(configs_target)

        configs_degrade.append(('PixelClip', {}))
        if debug:
            print('Final config:')
            print('\n'.join([str(c) for c in configs_degrade]))

        degrade_pipeline = ImageDegradationPipeline(configs_degrade)
        return degrade_pipeline, target_pipeline

    def __getitem__(self, index):
        if self.use_cache:
            try:
                data = self._load_tensor(index)
                return data
            except:
                # unsucessful at loading
                pass

        t0 = time()
        # original image
        target_path = os.path.join(self.data_root,
                                   self.file_list[index] + '.pth')
        # img = np.load(target_path).astype('float32')
        img = (np.array(Image.open(target_path)) / 255.0).astype(np.float32)
        # degradation pipeline, only one needing for N frame
        t1_load = time()
        degrade_param = self._randomize_parameter()
        degrade_pipeline, target_pipeline = self._create_pipeline(**{**self.pipeline_configs,
                                                                     **degrade_param})
        t2_create_pipeline = time()
        # Actually process image.
        img = FloatTensor(img).permute(2, 0, 1)

        # Crop first so that we don't waste computation on the whole image.
        # image with big jitter on original image
        img_big_jitter = bayer_crop_tensor(
            img, self.im_size_upscale, self.im_size_upscale, self.cropping
        )

        if len(img_big_jitter.size()) == 3:
            img_big_jitter = img_big_jitter.unsqueeze(0)

        # get N frames with big or small jitters

        burst_jitter = []
        for i in range(self.burst_length):
            # this is the ref. frame without shift
            if i == 0:
                burst_jitter.append(
                    F.interpolate(
                        img_big_jitter[:, :, self.big_restore_upscale:-self.big_restore_upscale,
                        self.big_restore_upscale:-self.big_restore_upscale],
                        scale_factor=1 / self.down_sample
                    )
                )
            else:
                # whether to flip the coin
                big_jitter = np.random.binomial(1, np.random.poisson(lam=1.5) / self.burst_length)
                if big_jitter:
                    burst_jitter.append(
                        F.interpolate(
                            bayer_crop_tensor(
                                img_big_jitter,
                                self.im_size_extra,
                                self.im_size_extra,
                                self.cropping
                            ),
                            scale_factor=1 / self.down_sample
                        )
                    )
                else:
                    img_small_jitter = img_big_jitter[:, :, self.big2small_upscale:-self.big2small_upscale,
                                       self.big2small_upscale:-self.big2small_upscale]
                    burst_jitter.append(
                        F.interpolate(
                            bayer_crop_tensor(
                                img_small_jitter,
                                self.im_size_extra,
                                self.im_size_extra,
                                self.cropping
                            ),
                            scale_factor=1 / self.down_sample
                        )
                    )
        burst_jitter = torch.cat(burst_jitter, dim=0)

        degraded = torch.zeros_like(burst_jitter)
        for i in range(self.burst_length):
            degraded[i, ...] = degrade_pipeline(burst_jitter[i, ...])
        # degraded = degrade_pipeline(target)
        target = burst_jitter[0, ...]

        # if not blind estimation, compute the estimated noise
        if not self.blind:
            read_sigma, poisson_k = degrade_param['read_noise_sigma'], degrade_param['poisson_k']
            noise = torch.sqrt(
                read_sigma ** 2 + poisson_k ** 2 * degraded[0, ...]
            ).unsqueeze(0)
            degraded = torch.cat([degraded, noise], dim=0)

        # If not exposure correction, also apply exposure adjustment to the image.
        if not self.pipeline_configs["exposure_correction"]:
            target = target_pipeline(target).squeeze()

        t3_degrade = time()
        exp_adjustment = degrade_param['exp_adjustment']
        # Bayer phase selection
        target = target.unsqueeze(0)

        im = torch.cat([degraded, target], 0)
        if self.pipeline_configs["bayer_crop_phase"] is None:
            # There are 4 phases of Bayer mosaick.
            phase = np.random.choice(4)
        else:
            phase = self.pipeline_configs["bayer_crop_phase"]
        x = phase % 2
        y = (phase // 2) % 2
        im = im[:, :, y:(y + self.im_size), x:(x + self.im_size)]
        degraded, target = torch.split(im, self.burst_length if self.blind else self.burst_length + 1, dim=0)
        t4_bayerphase = time()
        t5_resize = time()

        vis_exposure = 0 if self.pipeline_configs["exposure_correction"] else -exp_adjustment
        t6_bayermask = time()

        if DEBUG_TIME:
            # report
            print("--------------------------------------------")
            t_total = (t6_bayermask - t0) / 100.0
            t_load = t1_load - t0
            t_create_pipeline = t2_create_pipeline - t1_load
            t_process = t3_degrade - t2_create_pipeline
            t_bayercrop = t4_bayerphase - t3_degrade
            t_resize = t5_resize - t4_bayerphase
            t_bayermask = t6_bayermask - t5_resize
            print("load: {} ({}%)".format(t_load, t_load / t_total))
            print("create_pipeline: {} ({}%)".format(t_create_pipeline, t_create_pipeline / t_total))
            print("process: {} ({}%)".format(t_process, t_process / t_total))
            print("bayercrop: {} ({}%)".format(t_bayercrop, t_bayercrop / t_total))
            print("resize: {} ({}%)".format(t_resize, t_resize / t_total))
            print("bayermask: {} ({}%)".format(t_bayermask, t_bayermask / t_total))
            print("--------------------------------------------")

        data = {'degraded_img': degraded,
                'original_img': target.squeeze(),
                'vis_exposure': FloatTensor([vis_exposure]),
                }

        if self.use_cache:
            # TODO: Start a new thread to save.
            self._save_tensor(data, index)

        return data

    def __len__(self):
        return len(self.file_list)


class sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, num_samples):
        self.num_samples = num_samples
        self.total_num = len(data_source)

    def __iter__(self):
        if self.total_num % self.num_samples != 0:
            return iter(torch.randperm(self.total_num).tolist() + torch.randperm(self.total_num).tolist()[0:(
                                                                                                                    self.total_num // self.num_samples + 1) * self.num_samples - self.total_num])
        else:
            return iter(torch.randperm(self.total_num).tolist())


if __name__ == '__main__':
    # import argparse
    # from torch.utils.data import DataLoader
    #
    # parser = argparse.ArgumentParser(description='parameters for training')
    # parser.add_argument('--config_file', dest='config_file', default='kpn_specs/kpn_config.conf',
    #                     help='path to config file')
    # parser.add_argument('--config_spec', dest='config_spec', default='kpn_specs/configspec.conf',
    #                     help='path to config spec file')
    # parser.add_argument('--restart', action='store_true',
    #                     help='Whether to remove all old files and restart the training process')
    # parser.add_argument('--num_workers', '-nw', default=4, type=int, help='number of workers in data loader')
    # parser.add_argument('--num_threads', '-nt', default=8, type=int, help='number of threads in data loader')
    # parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    # parser.add_argument('--mGPU', '-m', action='store_true', help='whether to train on multiple GPUs')
    # args = parser.parse_args()
    #
    # print(args)
    #
    # config = read_config(args.config_file, args.config_spec)
    # train_config = config["training"]
    #
    #
    # i = 0
    # while i < 15:
    #     train_data = OnTheFlyDataset(train_config["dataset_configs"],
    #                                  use_cache=True,
    #                                  cache_dir='/home/bingo/burst-denoise/dataset/synthetic',
    #                                  blind=False,
    #                                  dataset_name='{:02d}'.format(i))
    #     train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=args.num_workers)
    #     for index, data in enumerate(train_loader):
    #         print('epoch {}, step {} is ok'.format(i, index))
    #     i += 1
    files = os.listdir('/home/bingo/burst-denoise/dataset/synthetic')
    files.sort()
    for index, f in enumerate(files):
        os.rename(os.path.join('/home/bingo/burst-denoise/dataset/synthetic', f),
                  os.path.join('/home/bingo/burst-denoise/dataset/synthetic', '{:06d}.pth'.format(index)))


