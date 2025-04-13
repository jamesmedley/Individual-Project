import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Load and preprocess image
img = Image.open('data/test/imgs/cju35eg0tdmjt085525sb4bua.jpg').convert('L')

img = ImageOps.pad(img, (256, 256))  # Resize to square with padding if needed
img_array = np.array(img).astype(np.float32)

# Define a Sobel vertical edge detection kernel
kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)


def normalise(img):
    return (img - img.min()) / (img.max() - img.min()) * 255


kernel_img = Image.fromarray(normalise(kernel).astype(np.uint8), mode='L')

# Upscale to 256x256 with nearest-neighbour interpolation
kernel_img_upscaled = kernel_img.resize((256, 256), resample=Image.NEAREST)

# Save
kernel_img_upscaled.save("kernel.png")
# Pad the kernel to the size of the image, and shift to centre
padded_kernel = np.zeros_like(img_array)
kh, kw = kernel.shape
padded_kernel[:kh, :kw] = kernel
padded_kernel = np.fft.ifftshift(padded_kernel)

# FFT of both image and kernel
F_img = np.fft.fft2(img_array)
F_kernel = np.fft.fft2(padded_kernel)

# Convolution in frequency domain
F_conv = F_img * F_kernel
conv_result = np.fft.fftshift(np.fft.ifft2(F_conv).real)

# Convert outputs to images
img_out = Image.fromarray(normalise(conv_result).astype(np.uint8))
img_orig = Image.fromarray(img_array.astype(np.uint8))

# Compute magnitude spectra for frequency domain visualisation
F_img_mag = np.log(np.abs(F_img) + 1)
F_kernel_mag = np.log(np.abs(F_kernel) + 1)
F_conv_mag = np.log(np.abs(F_conv) + 1)

# Convert to 8-bit images for saving
F_img_img = Image.fromarray(normalise(F_img_mag).astype(np.uint8))
F_kernel_img = Image.fromarray(normalise(F_kernel_mag).astype(np.uint8))
F_conv_img = Image.fromarray(normalise(F_conv_mag).astype(np.uint8))

# Save spatial domain results
img_orig.save('original.png')
img_out.save('fourier_convolution_output.png')

# Save frequency domain images
F_img_img.save('frequency_domain_image.png')
F_kernel_img.save('frequency_domain_kernel.png')
F_conv_img.save('frequency_domain_convolution_result.png')

# Display all together
plt.figure(figsize=(16, 8))

plt.subplot(2, 3, 1)
plt.imshow(img_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(kernel, cmap='gray')
plt.title('Sobel Kernel')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(conv_result, cmap='gray')
plt.title('Convolution via FFT (Spatial Domain)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(F_img_mag, cmap='gray')
plt.title('Image in Frequency Domain')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(F_kernel_mag, cmap='gray')
plt.title('Kernel in Frequency Domain')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(F_conv_mag, cmap='gray')
plt.title('Convolution in Frequency Domain')
plt.axis('off')

plt.tight_layout()
plt.savefig('fft_convolution_pipeline_with_frequency_domain.png')
plt.show()
