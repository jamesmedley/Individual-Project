import torch
import os
import pickle
from torch.utils.data import DataLoader
import torchvision.transforms as T
from utils.data_augment import JointTransform
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence, plot_objective, plot_evaluations
from unet import UNet
from utils.data_loading import BasicDataset
from evaluate import evaluate
from train import train_model  # Assuming the train_model function is imported from your script

# Define search space
space = [
    Real(1e-5, 1e-2, "log-uniform"),  # learning_rate
    Real(1e-10, 1e-2, "log-uniform"),  # weight_decay
    Real(0.5, 2.0)  # gradient_clipping
]

val_img_dir = './data/val/imgs/'
val_mask_dir = './data/val/masks/'
train_img_dir = './data/train/imgs/'
train_mask_dir = './data/train/masks/'

val_set = BasicDataset(val_img_dir, val_mask_dir)
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
    result = gp_minimize(objective, space, n_calls=150, random_state=42)

    with open('gp_minimize_result.pkl', 'wb') as f:  # save results for graphing later
        pickle.dump(result, f)

    # Best hyperparameters
    best_hyperparameters = {
        "Learning Rate": result.x[0],
        "Weight Decay": result.x[1],
        "Gradient Clipping": result.x[2]
    }

    # Print the best hyperparameters
    print("Best Hyperparameters:")
    for param, value in best_hyperparameters.items():
        print(f"{param}: {value}")

    # Save the best hyperparameters to a text file
    with open('best_hyperparameters.txt', 'w') as f:
        f.write("Best Hyperparameters:\n")
        for param, value in best_hyperparameters.items():
            f.write(f"{param}: {value}\n")

    plt.figure(figsize=(12, 8))
    plot_convergence(result)
    plt.savefig('convergence_plot.png', dpi=300)  # Save the plot as a PNG image
    plt.close()

    param_names = ["Learning Rate", "Weight Decay", "Gradient Clipping"]

    # Save objective function plot as an image
    plt.figure(figsize=(25, 25))
    plot_objective(result, dimensions=param_names, n_points=200, levels=30, size=5)
    plt.savefig('objective_plot.png', dpi=300)
    plt.close()

    # Save evaluations plot as an image
    plt.figure(figsize=(25, 25))
    plot_evaluations(result, dimensions=param_names, size=5)
    plt.savefig('evaluations_plot.png', dpi=300)
    plt.close()
