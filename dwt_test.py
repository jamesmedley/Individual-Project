import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_wavelets import DWTForward


def display_dwt_enhanced(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Convert to NCHW format

    # Apply 2D DWT
    xfm = DWTForward(J=1, mode='zero', wave='haar')  # Single level Haar wavelet transform
    Yl, Yh = xfm(img_tensor)

    # Extract LL, LH, HL, HH
    LL = Yl.squeeze().numpy()
    LH, HL, HH = torch.unbind(Yh[0], dim=2)  # Split along the third dimension
    LH, HL, HH = LH.squeeze().numpy(), HL.squeeze().numpy(), HH.squeeze().numpy()

    print("input:", img.shape)
    print("LL:", LL.shape)
    print("LH:", LH.shape)
    print("HL:", HL.shape)
    print("HH:", HH.shape)

    # Enhance visibility of edge coefficients
    def normalize_and_scale(coeff):
        coeff = np.abs(coeff)  # Take absolute value to avoid negative pixels
        return (coeff - coeff.min()) / (coeff.max() - coeff.min()) * 255  # Scale to [0,255]

    LH, HL, HH = normalize_and_scale(LH), normalize_and_scale(HL), normalize_and_scale(HH)

    # Display all images
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    axes[0].imshow(img, cmap='viridis')
    axes[0].set_title("Original Image")
    axes[1].imshow(LL, cmap='viridis')
    axes[1].set_title("LL (Approximation)")
    axes[2].imshow(LH, cmap='viridis')
    axes[2].set_title("LH (Horizontal)")
    axes[3].imshow(HL, cmap='viridis')
    axes[3].set_title("HL (Vertical)")
    axes[4].imshow(HH, cmap='viridis')
    axes[4].set_title("HH (Diagonal)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# Example usage
image_path = 'data/test/imgs/cju1dfeupuzlw0835gnxip369.jpg'  # Path to a sample image
image_path = 'Lenna_(test_image).png'
display_dwt_enhanced(image_path)