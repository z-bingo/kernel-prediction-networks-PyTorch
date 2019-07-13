""" I/O module

This unit deals with the nitty gritty of reading in DSLR raw camera and
various other formats.
"""
import numpy as np
import rawpy


def read_raw(path, n_bits=None):
    with rawpy.imread(path) as raw:
        im_ = raw.raw_image_visible.copy()

        # subtract black level
        im = np.zeros(im_.shape, dtype='float32')
        for i in range(len(raw.black_level_per_channel)):
            im += (im_ - raw.black_level_per_channel[i]) * (raw.raw_colors_visible == i).astype('float32')
        if n_bits is None:
            im /= np.amax(im)
        else:
            im /= np.power(2, n_bits)

        # shift bayer pattern
        red_idx = raw.color_desc.find(b'R')
        if red_idx == -1:
            print("Warning: Red is not in color description.")
            red_idx = 0

        raw_pattern = raw.raw_colors_visible[:8, :8].copy()
        red_pos = np.asarray(np.where(raw_pattern == red_idx))[:,0]
        row_offset = red_pos[0]
        # So that we start with GR
        col_offset = red_pos[1] + 1
        im = im[row_offset:, col_offset:]
        return im, \
               raw.rgb_xyz_matrix, \
               raw.camera_whitebalance
