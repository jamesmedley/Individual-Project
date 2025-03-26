import torch
import torchvision.transforms as transforms
from PIL import Image
from kymatio.torch import Scattering2D
import matplotlib.pyplot as plt
import math


def load_image(image_path, size=(128, 128)):
    """Loads an image and converts it to a tensor."""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def plot_coefficients(coeffs, title):
    """Plots all scattering coefficient images of a given order in a grid."""
    B, C, H, W = coeffs.shape  # Get batch, channels, height, width
    rows = math.ceil(math.sqrt(C))  # Compute number of rows for square layout
    cols = math.ceil(C / rows)  # Compute number of columns

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))  # Adjust size
    fig.suptitle(title, fontsize=16)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
        if i < C:  # If channel exists, plot it
            ax.imshow(coeffs[0, i].cpu().numpy(), cmap='gray')  # Show first batch (B=0)
            ax.set_title(f"Channel {i}")
        ax.axis('off')  # Hide axes for all subplots

    plt.tight_layout()
    plt.show()


def extract_scattering_coeffs(x, J=2, L=8):
    """
    Computes scattering coefficients and organises them by spatial resolution.

    Args:
        x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        J (int): Scale parameter for scattering transform.
        L (int): Number of orientations.

    Returns:
        list: Scattering coefficients as a list of dictionaries.
    """
    # Ensure input shape is (B, C, H, W)
    assert len(x.shape) == 4, "Input must be a 4D tensor (batch, channels, height, width)"

    # Get image size
    H, W = x.shape[-2:]

    # Define scattering transform with list output
    S = Scattering2D(J=J, shape=(H, W), L=L, out_type='array')

    # Compute scattering coefficients
    scattering_coeffs = S.scattering(x)

    B, C, S, H, W = scattering_coeffs.shape  # S = total number of scattering coefficients per channel

    # Compute index boundaries for orders
    n_order0 = 1  # Always 1 coefficient per channel
    n_order1 = J * L
    n_order2 = (L ** 2 * J * (J - 1)) // 2

    # Combine Order 0 and Order 1 into a single tensor
    coeffs_order1 = scattering_coeffs[:, :, :n_order0 + n_order1, :, :]  # (B, C, n_order0 + n_order1, H', W')
    coeffs_order2 = scattering_coeffs[:, :, n_order0 + n_order1:, :, :]  # (B, C, n_order2, H', W')

    # Reshape for CNN layers: Merge scattering channels into the channel dimension but keep RGB separate
    coeffs_order1 = coeffs_order1.reshape(B, C * (n_order0 + n_order1), H, W)  # (B, C * (n_order0 + n_order1), H', W')
    coeffs_order2 = coeffs_order2.reshape(B, C * n_order2, H, W)  # (B, C * n_order2, H', W')

    print(coeffs_order1.shape)
    print(coeffs_order2.shape)

    plot_coefficients(coeffs_order1, "order 1")
    plot_coefficients(coeffs_order2, "order 2")

    return scattering_coeffs


# Example usage:
image_path = "Lenna.png"  # Update with your image path
x = load_image(image_path)
coeffs = extract_scattering_coeffs(x, J=3, L=8)
print(coeffs.shape)


