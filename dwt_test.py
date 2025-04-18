import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
import torch
import torch.nn.functional as F


def haar_filter_bank(scale=1):
    """
    Generates Haar wavelet filters (LL, LH, HL, HH) for a given scale.
    Returns: dictionary of 2D filters
    """
    # Base low-pass and high-pass
    lp = np.array([1, 1]) / np.sqrt(2)
    hp = np.array([1, -1]) / np.sqrt(2)

    # Upsample by inserting zeros
    def upsample(filt, scale):
        up = np.zeros((2 ** scale - 1) * (len(filt) - 1) + len(filt))
        up[::2 ** scale] = filt
        return up

    # 1D filters
    lp_s = upsample(lp, scale)
    hp_s = upsample(hp, scale)

    # 2D separable filters
    LL = np.outer(lp_s, lp_s)
    LH = np.outer(lp_s, hp_s)
    HL = np.outer(hp_s, lp_s)
    HH = np.outer(hp_s, hp_s)

    return {'LL': LL, 'LH': LH, 'HL': HL, 'HH': HH}


def plot_haar_filter_bank(scales=[1, 2, 3], size=128):
    fig_spatial, axs_spatial = plt.subplots(len(scales), 4, figsize=(12, 8))
    fig_freq, axs_freq = plt.subplots(len(scales), 4, figsize=(12, 8))
    subbands = ['LL', 'LH', 'HL', 'HH']

    for i, scale in enumerate(scales):
        filters = haar_filter_bank(scale)
        for j, sb in enumerate(subbands):
            filt = filters[sb]

            # Pad to image size for frequency analysis
            padded = np.zeros((size, size))
            h, w = filt.shape
            padded[:h, :w] = filt

            # Spatial
            axs_spatial[i, j].imshow(padded, cmap='gray')
            axs_spatial[i, j].set_title(f'{sb}, scale {scale}')
            axs_spatial[i, j].axis('off')

            # Frequency
            fft_img = fftshift(np.abs(fft2(padded)))
            axs_freq[i, j].imshow(np.log(fft_img + 1), cmap='inferno')
            axs_freq[i, j].set_title(f'{sb}, scale {scale}')
            axs_freq[i, j].axis('off')

    fig_spatial.suptitle("Haar Filter Bank (Spatial Domain)", fontsize=16)
    fig_freq.suptitle("Haar Filter Bank (Fourier Domain)", fontsize=16)
    plt.tight_layout()
    plt.show()


# Run it
plot_haar_filter_bank()
