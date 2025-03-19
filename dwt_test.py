import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt


def display_dwt_enhanced(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)

    # Apply 2D DWT
    coeffs2 = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs2
    print("input: ", img.shape)
    print("LL: ", LL.shape)
    print("LH: ", LH.shape)
    print("HL: ", HL.shape)
    print("HH: ", HH.shape)

    # Enhance visibility of edge coefficients
    def normalize_and_scale(coeff):
        coeff = np.abs(coeff)  # Take absolute value to avoid negative pixels
        return (coeff - coeff.min()) / (coeff.max() - coeff.min()) * 255  # Scale to [0,255]

    LH = normalize_and_scale(LH)
    HL = normalize_and_scale(HL)
    HH = normalize_and_scale(HH)

    # Display all images
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    axes[0].imshow(img, cmap='viridis');
    axes[0].set_title("Original Image")
    axes[1].imshow(LL, cmap='viridis');
    axes[1].set_title("LL (Approximation)")
    axes[2].imshow(LH, cmap='viridis');
    axes[2].set_title("LH (Horizontal)")
    axes[3].imshow(HL, cmap='viridis');
    axes[3].set_title("HL (Vertical)")
    axes[4].imshow(HH, cmap='viridis');
    axes[4].set_title("HH (Diagonal)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# Example usage
image_path = 'data/test/imgs/cju1dfeupuzlw0835gnxip369.jpg'  # Path to a sample image
#image_path = 'Lenna_(test_image).png'  # Path to a sample image
display_dwt_enhanced(image_path)
