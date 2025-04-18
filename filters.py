import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from unet import UNet

# Create an impulse image (a single bright pixel in the centre)
def create_impulse_image(size=15):
    impulse = torch.zeros(1, 1, size, size)
    centre = size // 2
    impulse[0, 0, centre, centre] = 1.0
    return impulse

# Apply each kernel as a convolution and collect the output
def compute_impulse_responses(kernels, impulse):
    responses = []
    for i in range(kernels.shape[0]):
        kernel = kernels[i:i+1]  # shape: (1, in_channels, k, k)
        conv = nn.Conv2d(in_channels=kernel.shape[1], out_channels=1,
                         kernel_size=kernel.shape[2], bias=False)
        conv.weight.data = kernel
        conv.eval()
        with torch.no_grad():
            response = conv(impulse)
        responses.append(response)
    return torch.cat(responses, dim=0)

# Visualise responses in a grid
def plot_responses(responses, nrow=8, padding=2):
    grid = make_grid(responses, nrow=nrow, normalize=True, padding=padding)
    npimg = grid.cpu().numpy()
    plt.figure(figsize=(nrow, responses.shape[0] // nrow + 1))
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='viridis')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    model = UNet(n_channels=3, n_classes=1, bilinear=False)
    state_dict = torch.load("final_models/checkpoint_F10.4.pth", map_location='cpu')
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict, strict=False)
    else:
        model = state_dict
    model.eval()

    conv_layers = [layer for layer in model.modules()
                   if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d))]
    kernels = conv_layers[0].weight.data.clone()

    impulse = create_impulse_image(size=5)
    impulse = impulse.repeat(1, kernels.shape[1], 1, 1)  # Match input channels

    responses = compute_impulse_responses(kernels, impulse)
    plot_responses(responses)
