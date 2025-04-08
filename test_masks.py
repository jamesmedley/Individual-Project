import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from unet import UNet


# Load and prepare the model
def load_model(model_path):
    model = UNet(n_channels=3, n_classes=1)

    try:
        state_dict = torch.load(model_path, map_location='cpu')
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict, strict=False)
        else:
            model = state_dict
    except Exception as e:
        print("Failed to load model:", e)
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


def save_segmentation_mask(output_tensor, filename="mask1.png"):
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
    mask_img.save(filename)
    print("Segmentation mask image saved.")


def save_tensor_as_image(tensor, filename):
    # Ensure the tensor is in CPU and convert to numpy
    tensor = tensor.squeeze(0).cpu().detach().numpy()  # Remove batch dimension and convert to numpy
    tensor = np.transpose(tensor, (1, 2, 0))  # Convert from [C, H, W] to [H, W, C]

    # If the tensor is a float between 0 and 1, scale it to [0, 255]
    if tensor.max() <= 1.0:
        tensor = (tensor * 255).astype(np.uint8)
    else:
        tensor = tensor.astype(np.uint8)  # If it's already in [0, 255] range, just convert

    # Create a PIL image from the numpy array
    image = Image.fromarray(tensor)

    # Save the image
    image.save(filename)
    print(f"Image saved to {filename}")


def main():
    model_path = 'final_models/checkpoint_F11.pth'  # Path to your trained model
    image_path1 = "C:/Users/james/OneDrive/Documents/Computer Science/Y3/Individual Project - CM30082/Demo Test Polpys/Imgs/img1.png"
    image_path2 = "C:/Users/james/OneDrive/Documents/Computer Science/Y3/Individual Project - CM30082/Demo Test Polpys/Imgs/img2.png"
    image_path3 = "C:/Users/james/OneDrive/Documents/Computer Science/Y3/Individual Project - CM30082/Demo Test Polpys/Imgs/img3.png"
    image_path4 = "C:/Users/james/OneDrive/Documents/Computer Science/Y3/Individual Project - CM30082/Demo Test Polpys/Imgs/img4.png"

    # Load model and register hooks
    model = load_model(model_path)
    register_hooks(model)

    # Preprocess image and perform a forward pass
    image1, input_tensor1 = preprocess_image(image_path1)
    image2, input_tensor2 = preprocess_image(image_path2)
    image3, input_tensor3 = preprocess_image(image_path3)
    image4, input_tensor4 = preprocess_image(image_path4)

    with torch.no_grad():
        output_tensor1 = model(input_tensor1)
        output_tensor2 = model(input_tensor2)
        output_tensor3 = model(input_tensor3)
        output_tensor4 = model(input_tensor4)

    save_segmentation_mask(output_tensor1, filename="mask1.png")
    save_segmentation_mask(output_tensor2, filename="mask2.png")
    save_segmentation_mask(output_tensor3, filename="mask3.png")
    save_segmentation_mask(output_tensor4, filename="mask4.png")


if __name__ == "__main__":
    main()