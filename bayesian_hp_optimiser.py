import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from utils.data_augment import JointTransform
from skopt import gp_minimize
from skopt import dump
from skopt.space import Real
from unet import UNet
from utils.data_loading import BasicDataset
from evaluate import evaluate
from train import train_model  # Assuming the train_model function is imported from your script

# Define search space
space = [
    Real(1e-5, 1e-2, "log-uniform"),  # learning_rate
    Real(1e-8, 1e-2, "log-uniform"),  # weight_decay
    Real(0.5, 2.0)  # gradient_clipping
]

val_img_dir = './data/val/imgs/'
val_mask_dir = './data/val/masks/'
train_img_dir = './data/train/imgs/'
train_mask_dir = './data/train/masks/'

val_set = BasicDataset(val_img_dir, val_mask_dir, 0.5)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

image_transforms = [
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=(-45, 45)),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
]

transform = JointTransform(image_transforms=image_transforms)

train_set = BasicDataset(train_img_dir, train_mask_dir, 0.5, transform=transform)
n_train = len(train_set)

# 3. Create data loaders
loader_args = dict(batch_size=8, num_workers=20, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)


def objective(params):
    learning_rate, weight_decay, gradient_clipping = params
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # Initialize model
    model = UNet(n_channels=3, n_classes=1)  # Adjust n_classes as needed
    model.to(device)

    # Train model for a small number of epochs (to evaluate hyperparameters efficiently)
    train_model(
        model=model,
        device=device,
        train_loader=train_loader,
        n_train=n_train,
        train_set=train_set,
        epochs=50,
        batch_size=8,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clipping=gradient_clipping,
        save_checkpoint=False
    )

    # Evaluate model performance
    val_score = evaluate(model, val_loader, device, True)['dice_score'].item()
    return -val_score  # gp_minimize aims to minimize, so negate the score


if __name__ == '__main__':
    # Run the Bayesian Optimisation
    result = gp_minimize(objective, space, n_calls=100, random_state=42)
    dump(result, "hp_optim_results.pkl", store_objective=False)
