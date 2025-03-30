import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pytorch_wavelets import DTCWTForward, DTCWTInverse

def display_dtcwt_enhanced(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Convert to NCHW

    xfm = DTCWTForward(J=1)  # Use J=1 for one scale
    ifm = DTCWTInverse()

    # Create 12 outputs, one for the real and imaginary point spread functions
    # for each of the 6 orientations
    out = np.zeros((12, 128, 128))  # Array to hold all the outputs
    yl, yh = xfm(img_tensor)

    # Debugging: print the shapes of yl and yh
    print(f"Shape of yl: {len(yl)} scales")
    for i, l in enumerate(yl):
        print(f"Shape of yl[{i}]: {l.shape}")

    print(f"Shape of yh: {len(yh)} scales")
    for i, h in enumerate(yh):
        print(f"Shape of yh[{i}]: {h.shape}")

    # Manipulate the coefficients to enhance the image
    for b in range(6):  # 6 orientations
        for ri in range(2):  # real and imaginary parts
            yh[0][0, 0, b, 4, 4, ri] = 1  # Set the middle point for each orientation and real/imaginary
            out[b * 2 + ri] = ifm((yl, yh))  # Inverse transform after modifying coefficients
            yh[0][0, 0, b, 4, 4, ri] = 0  # Reset the middle point after processing

    # Plot the yl and yh coefficients
    fig, axes = plt.subplots(3, 6, figsize=(12, 6))

    # Plot yl (lowpass coefficients)
    for i in range(len(yl)):  # yl has length J (scales)
        # Check if yl[i] has 3 or 4 dimensions
        if len(yl[i].shape) == 3:
            axes[0, i].imshow(yl[i][0, :, :].detach().numpy(), cmap='gray')  # 3D tensor
        elif len(yl[i].shape) == 4:
            axes[0, i].imshow(yl[i][0, 0, :, :].detach().numpy(), cmap='gray')  # 4D tensor
        axes[0, i].set_title(f'Lowpass scale {i+1}')
        axes[0, i].axis('off')

    # Plot magnitude and phase of the highpass coefficients for each orientation
    for i in range(6):  # 6 orientations in DTCWT
        real_part = yh[0][0, 0, i, :, :, 0].detach().numpy()
        imag_part = yh[0][0, 0, i, :, :, 1].detach().numpy()

        # Compute magnitude and phase
        magnitude = np.sqrt(real_part ** 2 + imag_part ** 2)
        phase = np.angle(real_part + 1j * imag_part)

        # Plot magnitude
        axes[1, i].imshow(magnitude, cmap='jet')  # Jet colormap gives good visibility for magnitude
        axes[1, i].set_title(f'Highpass {i + 1} Magnitude')
        axes[1, i].axis('off')

    for i in range(6):  # 6 orientations in DTCWT
        real_part = yh[0][0, 0, i, :, :, 0].detach().numpy()
        imag_part = yh[0][0, 0, i, :, :, 1].detach().numpy()

        # Compute phase
        phase = np.angle(real_part + 1j * imag_part)

        # Plot phase
        axes[2, i].imshow(phase, cmap='twilight')  # Twilight colormap is good for phase visualization
        axes[2, i].set_title(f'Highpass {i + 1} Phase')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

    # Plot the 12 enhanced output images
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()

    for i in range(12):
        axes[i].imshow(out[i], cmap='gray')
        axes[i].set_title(f'Output {i+1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
image_path = 'Lenna_(test_image).png'
display_dtcwt_enhanced(image_path)
