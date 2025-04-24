import torch
import matplotlib.pyplot as plt
from pytorch_wavelets import DTCWTForward
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Load the image
image_path = 'retinal.jpg'  # Replace with the path to your image
img = Image.open(image_path).convert('L')  # Convert image to grayscale

# Resize image to be a power of 2 (necessary for DTCWT)
resize = transforms.Resize((512, 512))  # Resize the image to 512x512 for demonstration
img = resize(img)

# Convert the image to a tensor and add batch and channel dimensions
img_tensor = transforms.ToTensor()(img).unsqueeze(0)  # Shape: (1, 1, 512, 512)

# Apply DTCWT
xfm = DTCWTForward(J=1)  # J=1 for 1-level decomposition
yl, yh = xfm(img_tensor)
#  yl size: torch.Size([1, 128, 128])
#  yh size: torch.Size([1, 1, 6, 64, 64, 2]), 2 for complex parts

# Define the angles for the 6 orientations
angles = [15, 45, 75, 105, 135, 165]


# Function to compute and plot FFT of an image
def plot_fft(coeff, title, save_path):
    # Compute FFT and shift zero frequency to center
    coeff_fft = np.fft.fft2(coeff)
    coeff_fft_shifted = np.fft.fftshift(coeff_fft)

    # Compute the magnitude of the FFT
    coeff_fft_mag = np.abs(coeff_fft_shifted)

    # Plot and save the FFT magnitude
    plt.figure(figsize=(8, 8))
    plt.imshow(np.log(1 + coeff_fft_mag), cmap='gray', interpolation='nearest')  # Log scale for better visibility
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# Save input image and its FFT
input_image_filename = 'input_image.png'
plt.figure(figsize=(8, 8))
plt.imshow(np.array(img), cmap='gray', interpolation='nearest')
plt.axis('off')
plt.savefig(input_image_filename, dpi=300, bbox_inches='tight')
plt.close()

# Save FFT of the input image
input_fft_filename = 'input_fft.png'
plot_fft(np.array(img), 'Input Image', input_fft_filename)

# Save lowpass (yl) coefficient and its FFT
lowpass_filename = 'lowpass_coefficient.png'
plt.figure(figsize=(8, 8))
plt.imshow(yl[0, 0].detach().numpy(), cmap='viridis', interpolation='nearest')
plt.axis('off')
plt.savefig(lowpass_filename, dpi=300, bbox_inches='tight')
plt.close()

plot_fft(yl[0, 0].detach().numpy(), 'Lowpass Coefficient (yl)', 'lowpass_fft.png')

# Save each highpass magnitude and its FFT
for b in range(6):
    # Calculate the magnitude of the complex coefficients
    real_coeff = yh[0][0, 0, b, :, :, 0].detach().numpy()  # Real part
    imag_coeff = yh[0][0, 0, b, :, :, 1].detach().numpy()  # Imaginary part
    magnitude = np.sqrt(real_coeff ** 2 + imag_coeff ** 2)  # Magnitude calculation

    # File names for highpass coefficients and their FFTs
    highpass_filename = f'orientation_{angles[b]}_magnitude.png'
    fft_filename = f'orientation_{angles[b]}_fft.png'

    # Save the magnitude plot and its FFT
    plt.figure(figsize=(8, 8))
    plt.imshow(magnitude, cmap='viridis', interpolation='nearest')
    plt.axis('off')
    plt.savefig(highpass_filename, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot and save FFT of the highpass coefficient
    plot_fft(magnitude, f'Orientation {angles[b]} Coefficient Magnitude', fft_filename)
