import torch
import torchvision.transforms as transforms
from PIL import Image
from kymatio.torch import Scattering2D


def load_image(image_path, size=(128, 128)):
    """Loads an image and converts it to a tensor."""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


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
    S = Scattering2D(J=J, shape=(H, W), L=L, out_type='list')

    # Compute scattering coefficients
    scattering_coeffs = S.scattering(x)

    return scattering_coeffs


# Example usage:
image_path = "Lenna.png"  # Update with your image path
x = load_image(image_path)
coeffs = extract_scattering_coeffs(x, J=1, L=16)

# Loop through the list outputting meta information and tensor size
for i, coeff in enumerate(coeffs):
    coef_tensor = coeff['coef']
    j_values = coeff['j']
    n_values = coeff['n']
    theta_values = coeff['theta']
    print(f"Coefficient {i}: Size={coef_tensor.shape}, j={j_values}, n={n_values}, theta={theta_values}")
