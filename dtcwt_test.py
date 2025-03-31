import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from einops import rearrange


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

    #Yl = self.max_pool(Yl[0].unsqueeze(0))
    yl = yl[0].unsqueeze(0)
    yl = yl[:, :, ::2, ::2]

    yh = yh[0]  # Shape: (N, C, O, H, W, I)
    print(yh.shape)
    yh = rearrange(yh, 'b c o h w i -> b (c i o) h w')
    dtcwt_output = torch.cat([yl, yh], dim=1)
    print(dtcwt_output.shape)
    # Plot the features of xh (highpass coefficients after rearranging)
    fig, axes = plt.subplots(1, 13, figsize=(13, 8))

    for i in range(13):  # 6 orientations for xh
        # Accessing the real part of the coefficients for each orientation
        axes[i].imshow(dtcwt_output[0, i, :, :].detach().numpy(), cmap='gray')  # Real part of the coefficients
        axes[i].set_title(f'Highpass Real (xh[{i}])')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    quit()



    # Manipulate the coefficients to enhance the image
    for b in range(6):  # 6 orientations
        for ri in range(2):  # real and imaginary parts
            yh[0][0, 0, b, 32, 32, ri] = 1  # Set the middle point for each orientation and real/imaginary
            out[b * 2 + ri] = ifm((yl, yh))  # Inverse transform after modifying coefficients
            yh[0][0, 0, b, 32, 32, ri] = 0  # Reset the middle point after processing

    print(f"Shape of yl[0]: {yl[0].shape}")
    print(f"Shape of yh[0]: {yh[0].shape}")

    xl = yl[0].unsqueeze(0)
    xl = xl[:, :, ::2, ::2]
    print(f"Shape of xl after downsampling: {xl.shape}")

    xh = rearrange(yh[0], 'b c o h w i -> b (c i o) h w')

    # Debugging: print shapes to ensure correct rearrangement
    print(f"Shape of xl after rearrange: {xl.shape}")
    print(f"Shape of xh after rearrange: {xh.shape}")

    # Plot the features of xl (lowpass coefficients after rearranging)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(xl[0, 0, :, :].detach().numpy(), cmap='gray')
    axes[0].set_title('Lowpass Feature (xl[0, 0])')
    axes[0].axis('off')

    plt.tight_layout()
    plt.show()

    # Plot the features of xh (highpass coefficients after rearranging)
    fig, axes = plt.subplots(1, 12, figsize=(12, 8))

    for i in range(12):  # 6 orientations for xh
        # Accessing the real part of the coefficients for each orientation
        axes[i].imshow(xh[0, i, :, :].detach().numpy(), cmap='twilight')  # Real part of the coefficients
        axes[i].set_title(f'Highpass Real (xh[{i}])')
        axes[i].axis('off')


    plt.tight_layout()
    plt.show()

    # Plot the 12 enhanced output images (after manipulation)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.flatten()

    for i in range(12):
        axes[i].imshow(out[i], cmap='gray')
        axes[i].set_title(f'Output {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


    quit()

    # Debugging: print the shapes of yl and yh
    print(f"Shape of yl: {len(yl)} scales")
    for i, l in enumerate(yl):
        print(f"Shape of yl[{i}]: {l.shape}")

    print(f"Shape of yh: {len(yh)} scales")
    for i, h in enumerate(yh):
        print(f"Shape of yh[{i}]: {h.shape}")


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