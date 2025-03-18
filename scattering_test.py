import os
from pathlib import Path
from torch.utils.data import DataLoader
from kymatio.torch import Scattering2D
from utils.data_loading import BasicDataset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math

# Compute scattering transform
# Directories for pre-split datasets
train_img_dir = Path('./data/train/imgs/')
train_mask_dir = Path('./data/train/masks/')

if __name__ == '__main__':
    train_set = BasicDataset(train_img_dir, train_mask_dir, 1.0)

    x = train_set.__getitem__(498)["image"]
    print(x.shape)

    j = 1
    l = 8

    S1 = Scattering2D(J=j, shape=(x.shape[1], x.shape[1]), L=l)
    j = 2
    S2 = Scattering2D(J=j, shape=(x.shape[1], x.shape[1]), L=l)
    j = 3
    S3 = Scattering2D(J=j, shape=(x.shape[1], x.shape[1]), L=l)
    j = 4
    S4 = Scattering2D(J=j, shape=(x.shape[1], x.shape[1]), L=l)
    j = 5
    S5 = Scattering2D(J=j, shape=(x.shape[1], x.shape[1]), L=l)

    x1 = S1.scattering(x.contiguous())  # Shape: (C, scattering_channels, H', W')
    x2 = S2.scattering(x.contiguous())
    x3 = S3.scattering(x.contiguous())
    x4 = S4.scattering(x.contiguous())
    x5 = S5.scattering(x.contiguous())
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(x4.shape)
    print(x5.shape)
    quit()
    x1_resized = F.interpolate(x1, size=(256, 256), mode='bilinear', align_corners=False)
    print(x1_resized.shape)

    C, scattering_channels, H, W = x1_resized.shape

    # Number of coefficients to display
    num_coeffs_to_display = j + j * l + (l ** 2 * j * (j - 1)) // 2  # Change this number as needed

    # Calculate grid size (rows and columns) to fit the coefficients
    cols = 6  # Number of columns for coefficients (adjust as needed)
    rows = math.ceil(num_coeffs_to_display / cols)  # Calculate rows needed

    # Plot input image and selected scattering coefficients in a grid
    fig, axes = plt.subplots(rows + 1, cols, figsize=(12, 12))  # Extra row for the input image

    # Plot the input image with 3 channels (no need for a colormap)
    input_image = x.permute(1, 2, 0).numpy()  # Permute to (H, W, C) format for displaying
    axes[0, 0].imshow(input_image)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')

    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Plot the selected scattering coefficients in the remaining axes
    for i in range(num_coeffs_to_display - 1):
        coef = x1_resized[0, i, :, :].detach().numpy()  # Select scattering coefficient for the first image
        axes[i + 1].imshow(coef, cmap='viridis')  # Use 'viridis' for scattering coefficients
        axes[i + 1].set_title(f'Coeff {i + 1}')
        axes[i + 1].axis('off')

    # Hide any unused axes if num_coeffs_to_display is not a perfect fit for the grid
    for i in range(num_coeffs_to_display + 1, len(axes)):
        axes[i].axis('off')

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()
