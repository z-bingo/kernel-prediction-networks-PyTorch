import tifffile
import skimage
import numpy as np
import os
import argparse
import glob
import json
from tqdm import tqdm
from sklearn.feature_extraction.image import extract_patches_2d

import torch
from torch.autograd import Variable
from torch import FloatTensor

from data_generation.pipeline import ImageDegradationPipeline
from data_generation.constants import XYZ2sRGB, ProPhotoRGB2XYZ


def numpy2tensor(arr):
    if len(arr.shape) < 3:
        arr = np.expand_dims(arr, -1)
    return FloatTensor(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def tensor2numpy(t, idx=None):
    t = torch.clamp(t, 0, 1)
    if idx is None:
        t = t[0, ...]
    else:
        t = t[idx, ...]
    return t.permute(1, 2, 0).cpu().squeeze().numpy()


parser = argparse.ArgumentParser(description='')
parser.add_argument('--im_folder', required=True, help='path to input images')
parser.add_argument('--out_dir', required=True, help='path to place output')
parser.add_argument('--total_patch', type=int, required=True, help='total number of patches to generate')

parser.add_argument('--patch_per_image', type=int, default=5, help='Number of patch to generate from a single degradation of an image')
parser.add_argument('--patch_sz', type=int, default=256, help='Patch size (square patch for now)')
parser.add_argument('--fraction_train', type=float, default=0.8, help='Fraction of images to use as training')

parser.add_argument('--input_ext', default='tif', help='path to place output')
parser.add_argument('--max_exposure', type=float, default=0.0, help='maximum exposure adjustment in stops')
parser.add_argument('--min_exposure', type=float, default=0.0, help='minimum exposure adjustment in stops')
parser.add_argument('--max_gaussian_noise', type=float, default=0.0, help='maximum gaussian noise std (on range 0 - 1)')
parser.add_argument('--min_gaussian_noise', type=float, default=0.0, help='minimum gaussian noise std (on range 0 - 1)')
parser.add_argument('--max_poisson_noise', type=float, default=0.0, help='maximum poisson noise mult (See image_processing.PoissonNoise for detail)')
parser.add_argument('--min_poisson_noise', type=float, default=0.0, help='minimum poisson noise mult (See image_processing.PoissonNoise for detail)')
parser.add_argument('--skip_degraded', action="store_true", help='Whether to skip degraded images.')
parser.add_argument('--dwn_factor', type=float, default=4, help='Factor to downsample.')
args = parser.parse_args()


im_names = glob.glob(os.path.join(args.im_folder, '*.' + args.input_ext))
im_names = sorted([os.path.basename(i) for i in im_names])

# Create output folder
os.makedirs(args.out_dir, exist_ok=True)
train_dir = os.path.join(args.out_dir, 'train')
test_dir = os.path.join(args.out_dir, 'test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
for base_dir in [train_dir, test_dir]:
    target_dir = os.path.join(base_dir, 'images', 'target')
    degraded_dir = os.path.join(base_dir, 'images', 'degraded')
    meta_dir = os.path.join(base_dir, 'meta')
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(degraded_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

n_count = 0
img_idx = 0

progress_bar = tqdm(total=args.total_patch)
while n_count < args.total_patch:
    if img_idx < args.fraction_train * len(im_names):
        base_dir = train_dir
    else:
        base_dir = test_dir

    target_dir = os.path.join(base_dir, 'images', 'target')
    degraded_dir = os.path.join(base_dir, 'images', 'degraded')
    meta_dir = os.path.join(base_dir, 'meta')

    name = im_names[img_idx]
    path = os.path.join(args.im_folder, name)
    # We know 5k dataset is 16-bits.
    raw_im = tifffile.imread(path).astype('float32') / 65536.0
    raw_im = FloatTensor(raw_im).permute(2, 0, 1).unsqueeze(0)

    # Define pipeline
    poisson_k = np.random.uniform(args.min_poisson_noise, args.max_poisson_noise)
    read_noise_sigma = np.random.uniform(args.min_gaussian_noise, args.max_gaussian_noise)
    dwn_factor = args.dwn_factor
    exp_adjustment = np.random.uniform(args.min_exposure, args.max_exposure)
    configs_prepreprocess = [
                ('UndoProPhotoRGBGamma', {}),
                # Convert to sRGB
                ('ColorSpaceConversionMatrix', {'matrix': torch.matmul(XYZ2sRGB, ProPhotoRGB2XYZ)}),
        ]

    configs_preprocess = [
        # Blur and downsample to reduce noise
        ('GaussianBlur', {'sigma_x': dwn_factor}),
        ('PytorchResizing', {'resizing_factor': 1.0/dwn_factor, 'mode': 'nearest'})
    ]
    configs_degrade = [
        ('ExposureAdjustment', {'nstops': exp_adjustment}),
        # ('MotionBlur', {'amt': [3, 2], 'direction': [0, 45,]}),
        ('BayerMosaicking', {}),
        # Add artificial noise.
        ('PoissonNoise',{'sigma': FloatTensor([poisson_k] * 3), 'mosaick_pattern': 'bayer'}),
        ('GaussianNoise',{'sigma': FloatTensor([read_noise_sigma] * 3), 'mosaick_pattern': 'bayer'}),
        ('PixelClip', {}),
        ('ExposureAdjustment', {'nstops': -exp_adjustment}),
        ('PixelClip', {}),
        ('NaiveDemosaicking', {}),
        ('PixelClip', {}),
    ]
    configs_denoise = [
            ('DenoisingBilateral',{'sigma_s': 1.0, 'sigma_r': 0.1}),
            ('PixelClip', {}),
            ('sRGBGamma', {}),
        ]

    pipeline_prepreprocess = ImageDegradationPipeline(configs_prepreprocess)
    pipeline_preprocess = ImageDegradationPipeline(configs_preprocess)
    pipeline_degrade = ImageDegradationPipeline(configs_degrade)
    pipeline_denoise = ImageDegradationPipeline(configs_denoise)


    demosaicked = pipeline_prepreprocess(raw_im)
    preprocessed = pipeline_preprocess(demosaicked)
    degraded = pipeline_degrade(preprocessed)
    denoised = pipeline_denoise(degraded)

    denoised_numpy = tensor2numpy(denoised)
    preprocessed_numpy = tensor2numpy(preprocessed)
    stacked = np.concatenate((denoised_numpy, preprocessed_numpy), axis=-1)
    patches = extract_patches_2d(stacked,
                                 (args.patch_sz, args.patch_sz),
                                 args.patch_per_image)
    degraded_patches, target_patches = np.split(patches, 2, axis=-1)

    target_patches = np.split(target_patches, target_patches.shape[0])
    degraded_patches = np.split(degraded_patches, degraded_patches.shape[0])

    meta = dict(orig=name,
                poisson_k=poisson_k,
                read_noise_sigma=read_noise_sigma,
                exp_adjustment=exp_adjustment,
                dwn_factor=dwn_factor)
    n_patches = len(degraded_patches)
    for i in range(n_patches):
        patch_idx = n_count + i + 1

        degraded = np.clip(degraded_patches[i] * 255.0, 0, 255).astype('uint8')
        if not args.skip_degraded:
            skimage.io.imsave(os.path.join(degraded_dir,
                "{:06d}.png".format(patch_idx)
                                      ),
                              np.squeeze(degraded))
        np.save(os.path.join(target_dir,
            "{:06d}.npy".format(patch_idx)
                            ),
                np.squeeze(target_patches[i]))
        with open(os.path.join(meta_dir,
            '{:06d}.json'.format(patch_idx)),
                               'w') as f:
            json.dump(meta, f)
    n_count += n_patches
    img_idx = (img_idx + 1) % len(im_names)
    progress_bar.update(n_patches)
progress_bar.close()
