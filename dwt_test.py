import numpy as np
import torch
from pytorch_wavelets import DWTForward, DWTInverse
import matplotlib.pyplot as plt

# Initialize forward and inverse transform
xfm = DWTForward(J=3, wave='db1', mode='zero')  # Use Daubechies 1 wavelet (Haar)
ifm = DWTInverse(wave='db1', mode='zero')

# Input tensor (64x64 image of zeros)
x = torch.zeros(1, 1, 64, 64)

# Create a list to store the outputs for each decomposition level
out = np.zeros((4, 64, 64))

# Perform the forward transform
yl, yh = xfm(x)  # yl is the LL band, yh is a list of high-pass bands

# Modify and inverse transform for each level of decomposition
for level in range(3):  # DWT has 3 levels of decomposition (scales)
    for ri in range(3):  # LH (horizontal), HL (vertical), HH (diagonal) for each scale
        yh[level][0, 0, ri, 4, 4] = 1  # Set impulse at (4, 4) in the coefficient
        out[level] = ifm((yl, yh))  # Perform inverse transform with modified yh
        yh[level][0, 0, ri, 4, 4] = 0  # Reset the impulse to 0

# Plot the outputs (visualising the inverse transform results)
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
axes = axes.flatten()

for i in range(4):
    axes[i].imshow(out[i], cmap='gray')
    axes[i].set_title(f'Output {i+1}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
