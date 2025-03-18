import torch
import torch.nn as nn
import torch.nn.functional as F
from kymatio.torch import Scattering2D
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import cm
import numpy as np
from unet import ScatUNet
from unet import UNet


# Load and prepare the model
def load_model(model_path):
    model = ScatUNet(n_channels=3, n_classes=1, bilinear=False, J=1, L=16, input_shape=(128, 128))

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(state_dict, dict):  # If it's a state_dict, load it
        model.load_state_dict(state_dict, strict=False)
    else:  # If the entire model is saved, load it directly
        model = state_dict
    model.eval()  # Set model to evaluation mode
    return model


# Hook function to extract feature maps
feature_maps = {}


def hook_fn(module, input, output):
    feature_maps[module] = output


# Register hooks to convolutional layers
def register_hooks(model):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):  # Register only for Conv2D layers
            layer.register_forward_hook(hook_fn)


# Preprocess the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to the desired input size
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image, input_tensor


def plot_feature_maps(feature_maps):
    for module, fmap in feature_maps.items():
        fmap = fmap.squeeze(0)  # Remove batch dimension
        num_channels = fmap.shape[0]

        # Plot a subset of feature maps (e.g., first 8 channels)
        num_plots = min(num_channels, 8)
        fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))

        # Ensure axes is always an iterable (even if there's one plot)
        if num_plots == 1:
            axes = np.array([axes])

        for i, ax in enumerate(axes):
            ax.imshow(fmap[i].cpu().numpy(), cmap='viridis')
            ax.axis('off')

        plt.suptitle(f"Feature Maps from Layer: {module}")
        plt.show()


def plot_filters(model, layer_indices=None):
    """
    Plot filters of convolutional layers from the model.

    Args:
        model (nn.Module): The trained model.
        layer_indices (list, optional): Specific layer indices to visualise.
                                        If None, visualises all convolutional layers.
    """
    conv_layers = [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]

    if layer_indices is not None:
        conv_layers = [conv_layers[i] for i in layer_indices]

    for idx, conv_layer in enumerate(conv_layers):
        filters = conv_layer.weight.data.cpu().numpy()  # Shape: (out_channels, in_channels, H, W)

        num_filters = filters.shape[0]  # Number of filters
        grid_size = int(np.ceil(np.sqrt(num_filters)))  # Determine grid size for plotting

        # Create a grid of subplots
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))

        for i, ax in enumerate(axes.flat):
            if i < num_filters:
                # Visualise the first input channel of each filter
                filter_img = filters[i, 0, :, :]  # Shape: (H, W)
                ax.imshow(filter_img, cmap='viridis')
                ax.axis('off')
            else:
                ax.axis('off')  # Hide extra axes

        plt.suptitle(f"Filters from Layer {idx + 1}")
        plt.show()


def save_feature_maps_as_png(feature_maps, output_path="feature_maps.png"):
    """
    Save feature maps from each layer as a high-resolution PNG, stacked vertically.

    Args:
        feature_maps (dict): Dictionary with layer names as keys and feature maps as values.
        output_path (str): Path to save the output PNG file.
    """
    # Count the number of layers and determine layout
    num_layers = len(feature_maps)
    num_features = 8  # Display the first 8 features for each layer

    # Set up figure dimensions
    fig_height = num_layers * 2  # Adjust height based on the number of layers
    fig_width = num_features * 2
    fig, axes = plt.subplots(num_layers, num_features, figsize=(fig_width, fig_height))

    # If there's only one layer, make `axes` iterable
    if num_layers == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, (module, fmap) in enumerate(feature_maps.items()):
        fmap = fmap.squeeze(0)  # Remove batch dimension
        num_channels = fmap.shape[0]

        # Loop through the first 8 feature maps
        for col_idx in range(num_features):
            ax = axes[row_idx, col_idx]
            if col_idx < num_channels:
                feature = fmap[col_idx].cpu().numpy()  # Convert to NumPy
                ax.imshow(feature, cmap="viridis")
            ax.axis("off")  # Turn off axis for a clean look

        # Label each row with the layer name
        axes[row_idx, 0].set_ylabel(module, rotation=0, labelpad=40, fontsize=10)

    # Remove spacing between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Feature maps saved to {output_path}")


def save_channel_images(image, input_tensor):
    """ Save separate images for each color channel (RGB) side by side with no gaps. """
    # Extract individual channels (RGB)
    r, g, b = input_tensor.squeeze(0)[0], input_tensor.squeeze(0)[1], input_tensor.squeeze(0)[2]

    # Convert each channel to a numpy array in the range [0, 255]
    r_img = (r.cpu().numpy() * 255).astype(np.uint8)
    g_img = (g.cpu().numpy() * 255).astype(np.uint8)
    b_img = (b.cpu().numpy() * 255).astype(np.uint8)

    # Stack the channels horizontally
    rgb_image = np.concatenate((r_img, g_img, b_img), axis=1)

    # Convert to a PIL image
    rgb_image_pil = Image.fromarray(rgb_image)

    # Save the combined image
    rgb_image_pil.save("feature_visualisation/channels_combined.png")

    print("RGB channels saved as one image.")


def save_segmentation_mask(output_tensor):
    """ Save the segmentation mask from the model's output. """
    # Remove batch dimension (assuming output shape is [batch_size, 1, height, width])
    mask = output_tensor.squeeze(0)  # Shape becomes [1, height, width]

    # If the model produces a single-channel output (for binary segmentation)
    if mask.shape[0] == 1:
        mask = mask[0]  # Remove the channel dimension

    # Apply a threshold to get a binary mask
    mask = (mask > 0.5).cpu().numpy()  # Threshold to create a binary mask

    # Convert to PIL image
    mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")  # Convert to 8-bit image
    mask_img.save("feature_visualisation/segmentation_mask.png")
    print("Segmentation mask image saved.")


def save_scattering_coefficients(input_tensor, J=1, L=16, input_shape=(128, 128)):
    # Scattering transform
    S = Scattering2D(J=J, shape=input_shape, L=L)
    scattering_channels = 1 + L * J + (L ** 2 * J * (J - 1)) // 2
    scattering_coeffs = S.scattering(input_tensor.contiguous())

    B, C, scattering_channels, H, W = scattering_coeffs.shape
    scattering_coeffs = scattering_coeffs.view(B, -1, H, W)
    scattering_coeffs_resized = F.interpolate(scattering_coeffs, size=input_shape, mode='bilinear', align_corners=False)

    # Convert the coefficients into a numpy array (removing batch dimension)
    scattering_coeffs_resized = scattering_coeffs_resized.squeeze(0).cpu().numpy()

    # Get number of coefficients
    num_coeffs = scattering_coeffs_resized.shape[0]

    # Calculate the number of rows and columns for a square grid
    grid_size = int(np.ceil(np.sqrt(num_coeffs)))  # Find the nearest square grid size
    num_rows = grid_size
    num_cols = grid_size

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols * 2, num_rows * 2))

    # Apply viridis colormap and save the coefficients in the grid
    for i in range(num_coeffs):
        row = i // num_cols
        col = i % num_cols
        coeff_img = scattering_coeffs_resized[i]

        # Normalize to [0, 1]
        coeff_img = (coeff_img - coeff_img.min()) / (coeff_img.max() - coeff_img.min())

        # Apply viridis colormap
        coeff_img = cm.viridis(coeff_img)  # Apply viridis colormap
        coeff_img = (coeff_img[:, :, :3] * 255).astype(np.uint8)  # Convert to RGB

        # Plot the coefficient image
        axes[row, col].imshow(coeff_img)
        axes[row, col].axis('off')

    # Remove any unused axes if the grid is not completely filled
    for i in range(num_coeffs, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axes[row, col].axis('off')

    # Save the scattering coefficients image
    plt.tight_layout()
    plt.savefig("feature_visualisation/scattering_coefficients.png", dpi=300)
    plt.close(fig)
    print("Scattering coefficients grid (square) image saved.")


def main():
    model_path = './final_models/8/checkpoint_epoch50.pth'  # Path to your trained model
    image_path = 'data/test/imgs/cju1dfeupuzlw0835gnxip369.jpg'  # Path to a sample image

    # Load model and register hooks
    model = load_model(model_path)
    register_hooks(model)

    # Preprocess image and perform a forward pass
    image, input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output_tensor = model(input_tensor)  # Forward pass through the model

    save_channel_images(image, input_tensor)
    save_segmentation_mask(output_tensor)
    save_feature_maps_as_png(feature_maps, output_path="feature_visualisation/feature_maps_highres.png")
    save_scattering_coefficients(input_tensor)
    # Visualise feature maps
    # plot_feature_maps(feature_maps)
    # Visualise learned filters
    # plot_filters(model)

if __name__ == "__main__":
    main()
