import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_wavelets import DWTForward
import pywt
from scipy.signal import convolve2d
from PIL import Image


def display_dwt_enhanced(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Convert to NCHW format

    # Apply 2D DWT
    xfm = DWTForward(J=1, mode='zero', wave='haar')  # Single level Haar wavelet transform
    Yl, Yh = xfm(img_tensor)
    print(img_tensor.shape)
    print(Yl.shape)
    print(Yh[0].shape)
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


def gaussian(size, sigma=1.5, shift_x=0, shift_y=0):
    """Generates a 2D Gaussian kernel centered in the middle, with a shift."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)

    # Shift the Gaussian kernel by shifting the indices
    xx_shifted = xx + shift_x
    yy_shifted = yy + shift_y

    # Apply the Gaussian function after shifting
    gaussian_kernel = np.exp(-(xx_shifted ** 2 + yy_shifted ** 2) / (2 * sigma ** 2))

    # Normalize the Gaussian kernel
    return gaussian_kernel / np.sum(gaussian_kernel)


# Define a 2D Haar wavelet filter
def haar_wavelet_2d(size):
    # Haar wavelet for 2D: basic wavelet on x and y directions
    low_pass = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    high_pass = np.array([1/np.sqrt(2), -1/np.sqrt(2)])

    # 2D tensor product to create the separable 2D wavelet
    LL = np.outer(low_pass, low_pass)  # Low-Low filter
    LH = np.outer(low_pass, high_pass)  # Low-High filter
    HL = np.outer(high_pass, low_pass)  # High-Low filter
    HH = np.outer(high_pass, high_pass)  # High-High filter

    # Use np.tile to correctly extend the wavelets to the desired size
    LL_resized = np.tile(LL, (size // 2, size // 2))  # 2x2 tiling for LL
    LH_resized = np.tile(LH, (size // 2, size // 2))  # 2x2 tiling for LH
    HL_resized = np.tile(HL, (size // 2, size // 2))  # 2x2 tiling for HL
    HH_resized = np.tile(HH, (size // 2, size // 2))  # 2x2 tiling for HH

    gaussian_kernel = gaussian(size)

    LL_resized = LL_resized * gaussian_kernel
    LH_resized = LH_resized * gaussian_kernel
    HL_resized = HL_resized * gaussian_kernel
    HH_resized = HH_resized * gaussian_kernel


    # Plot the 2D Haar wavelet components (LL, LH, HL, HH)
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plot each component
    axs[0, 0].imshow(LL_resized, cmap='gray', interpolation='nearest')
    axs[0, 0].set_title('LL (Low-Low)')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(LH_resized, cmap='gray', interpolation='nearest')
    axs[0, 1].set_title('LH (Low-High)')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(HL_resized, cmap='gray', interpolation='nearest')
    axs[1, 0].set_title('HL (High-Low)')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(HH_resized, cmap='gray', interpolation='nearest')
    axs[1, 1].set_title('HH (High-High)')
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()



# List of all wavelet names
wavelet_names = [
    'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10',
    'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20',
    'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10',
    'coif1', 'coif2', 'coif3', 'coif4', 'coif5',
    'morl', 'cmor', 'dmey', 'shan', 'gaus1', 'gaus2', 'gaus3', 'gaus4'
]


# Function to plot wavelet filters
def plot_wavelet_filters(wavelet_names):
    # Number of wavelets to plot
    num_wavelets = len(wavelet_names)

    # Determine the grid size for the subplots
    ncols = 10  # Define a fixed number of columns
    nrows = (num_wavelets + ncols - 1) // ncols  # Calculate the number of rows needed

    # Create a figure with the required grid size
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))

    # Flatten axes array for easier iteration
    axes = axes.flatten()

    # Loop through all wavelet names
    for i, wavelet_name in enumerate(wavelet_names):
        try:
            # Check if it's a continuous wavelet (e.g., morl, cmor)
            if wavelet_name in ['morl', 'cmor', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'shan']:
                # For continuous wavelets, use ContinuousWavelet
                wavelet = pywt.ContinuousWavelet(wavelet_name)
                # Continuous wavelets return only two values (psi and x)
                psi, x = wavelet.wavefun(level=4)
                # Plot the real part of the wavelet function
                v = np.outer(psi, psi)
                axes[i].imshow(v, cmap='seismic', aspect='equal', vmin=-np.max(np.abs(v)), vmax=np.max(np.abs(v)))
                axes[i].set_title(f'{wavelet_name} (Real)')
                axes[i].axis('off')
            else:
                # For discrete wavelets, use Wavelet
                wavelet = pywt.Wavelet(wavelet_name)
                # Discrete wavelets return three values (phi, psi, x)
                phi, psi, x = wavelet.wavefun(level=4)
                # Plot the real part of the outer product of the wavelet function
                v = np.outer(psi, psi)
                axes[i].imshow(v, cmap='seismic', aspect='equal', vmin=-np.max(np.abs(v)), vmax=np.max(np.abs(v)))
                axes[i].set_title(f'{wavelet_name}')
                axes[i].axis('off')
        except ValueError as e:
            # Catch ValueError if wavefun() does not return the expected number of values
            print(f"Wavelet {wavelet_name} not properly unpacked, skipping. Error: {e}")
            axes[i].axis('off')
            continue
        except Exception as e:
            # Catch other unexpected errors (e.g., invalid wavelet name)
            print(f"Unexpected error with wavelet {wavelet_name}: {e}")
            axes[i].axis('off')
            continue

    # Adjust layout to avoid overlapping
    plt.tight_layout()
    plt.show()


# Plot the filters for all wavelets
plot_wavelet_filters(wavelet_names)


# Example usage
image_path = 'data/test/imgs/cju1dfeupuzlw0835gnxip369.jpg'  # Path to a sample image
image_path = 'Lenna_(test_image).png'
#display_dwt_enhanced(image_path)
