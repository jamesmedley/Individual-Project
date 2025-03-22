import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pytorch_wavelets import DTCWTForward


def display_dtcwt_enhanced(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Convert to NCHW

    # Apply 2D DTCWT
    xfm = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')  # Single-level DTCWT
    Yl, Yh = xfm(img_tensor)  # Yh is a tuple, one per level

    # Extract real and imaginary parts
    Yh_real = Yh[0][..., 0].squeeze(0).squeeze(0).numpy()  # Shape: (6, H, W)
    Yh_imag = Yh[0][..., 1].squeeze(0).squeeze(0).numpy()

    # Compute magnitude and phase
    magnitude = np.sqrt(Yh_real ** 2 + Yh_imag ** 2)
    phase = np.arctan2(Yh_imag, Yh_real)  # Phase in radians

    # Enhance for visualisation
    def normalize_and_scale(coeff):
        coeff = np.abs(coeff)
        return (coeff - coeff.min()) / (coeff.max() - coeff.min()) * 255  # Scale to [0,255]

    mag_vis = normalize_and_scale(magnitude)
    phase_vis = (phase + np.pi) / (2 * np.pi) * 255  # Scale phase to [0,255]

    # Display results
    fig, axes = plt.subplots(4, 7, figsize=(15, 10))
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("Original Image")

    # Labels for the subbands (DTCWT has 6 orientations)
    angles = ["15°", "45°", "75°", "105°", "135°", "165°"]

    for i in range(6):
        axes[0, i + 1].imshow(Yh_real[i], cmap='gray')
        axes[0, i + 1].set_title(f"Real {angles[i]}")
        axes[1, i + 1].imshow(Yh_imag[i], cmap='gray')
        axes[1, i + 1].set_title(f"Imag {angles[i]}")
        axes[2, i + 1].imshow(mag_vis[i], cmap='inferno')
        axes[2, i + 1].set_title(f"Magnitude {angles[i]}")

    axes[2, 0].imshow(phase_vis.mean(axis=0), cmap='twilight')
    axes[2, 0].set_title("Phase (Averaged)")
    axes[3, 0].imshow(Yl[0][0], cmap='gray')
    axes[3, 0].set_title("Yl")

    for ax in axes.flatten():
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# Example usage
image_path = 'Lenna_(test_image).png'
display_dtcwt_enhanced(image_path)
