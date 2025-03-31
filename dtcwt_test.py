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
    X = torch.randn(8, 256, 64, 64)

    out = np.zeros((12, 128, 128))  # Array to hold all the outputs
    yl, yh = xfm(X)

    print(yl.shape)

    yl = yl[:, :, ::2, ::2]

    yh = yh[0]  # Shape: (N, C, O, H, W, I)
    print(yl.shape)
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
   # plt.show()

    print("RECONSTRUCTING")
    batch, channels, height, width = dtcwt_output.shape
    num_base_channels = channels // 16
    Yl, Yh = torch.split(dtcwt_output, [4*num_base_channels, 12*num_base_channels], dim=1)
    print(Yl.shape)
    print(Yh.shape)
    Yl = rearrange(Yl, 'b c h w -> b (4 c) (2 h) (2 w)')
    print(Yl.shape)
    Yh = rearrange(Yh, 'b (12 c) h w -> b 2 6 c h w')
    print(Yh.shape)
    # Perform inverse DTCWT
    x = ifm((Yl, Yh))
    print(x.shape)



# Example usage
image_path = 'Lenna_(test_image).png'
display_dtcwt_enhanced(image_path)
