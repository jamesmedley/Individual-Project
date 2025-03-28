import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import scipy.stats as stats  # Added for Bayesian credible intervals
from utils.data_augment import JointTransform
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np  # Added for numerical operations

from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from evaluate import evaluate

# Directories
train_img_dir = Path('./data/train/imgs/')
train_mask_dir = Path('./data/train/masks/')
test_img_dir = Path('./data/test/imgs/')
test_mask_dir = Path('./data/test/masks/')
dir_checkpoint = Path('./checkpoints/')


def compute_credible_interval(successes, failures, confidence=0.95):
    """Compute the Bayesian credible interval using a Beta distribution."""
    alpha = successes + 1  # Prior Beta(1,1)
    beta = failures + 1
    lower_bound = stats.beta.ppf((1 - confidence) / 2, alpha, beta)
    upper_bound = stats.beta.ppf(1 - (1 - confidence) / 2, alpha, beta)
    return lower_bound, upper_bound


def train_and_test_model(model, device, epochs, batch_size, learning_rate, img_scale, weight_decay, momentum,
                         gradient_clipping, max_repeats, results_file):
    image_transforms = [
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=(-45, 45)),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ]

    transform = JointTransform(image_transforms=image_transforms)
    train_set = BasicDataset(train_img_dir, train_mask_dir, img_scale, transform=transform)
    n_train = len(train_set)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=10,
                              pin_memory=True)

    test_set = BasicDataset(test_img_dir, test_mask_dir, img_scale)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1, num_workers=10, pin_memory=True,
                             drop_last=True)

    # Bayesian tracking of metrics
    metric_names = ['dice_score', 'mIoU', 'precision', 'recall']
    successes = {name: 0 for name in metric_names}
    failures = {name: 0 for name in metric_names}

    for run in range(max_repeats):
        model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        use_amp = torch.cuda.is_available()
        grad_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

        logging.info(f'Starting training run {run + 1}/{max_repeats}')

        for epoch in range(1, epochs + 1):
            model.train()
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                for batch in train_loader:
                    images, true_masks = batch['image'], batch['mask']

                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    true_masks = true_masks.to(device=device, dtype=torch.long)

                    with torch.autocast(device.type if device.type != 'mps' else 'cpu',
                                        enabled=torch.cuda.is_available()):
                        masks_pred = model(images)
                        if model.n_classes == 1:
                            loss = criterion(masks_pred.squeeze(1), true_masks.float())
                            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        else:
                            loss = criterion(masks_pred, true_masks)
                            loss += dice_loss(
                                F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    pbar.update(images.shape[0])
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

        # Evaluate the model
        test_results = evaluate(model, test_loader, device, torch.cuda.is_available())

        # Update Bayesian tracking
        for metric_name in metric_names:
            if metric_name in test_results:  # Ensure the metric exists
                value = test_results[metric_name]
                if isinstance(value, torch.Tensor):  # Convert tensor to float if needed
                    value = value.item()
                successes[metric_name] += value * len(test_loader)
                failures[metric_name] += (1 - value) * len(test_loader)
            else:
                logging.warning(f'Metric "{metric_name}" not found in test results.')

        # Compute credible intervals
        credible_intervals = {}
        for metric_name in metric_names:
            lb, ub = compute_credible_interval(successes[metric_name], failures[metric_name])
            credible_intervals[metric_name] = (lb, ub)

        with open(results_file, 'a') as f:
            f.write(f'Run {run + 1}: {test_results}, Credible Intervals: {credible_intervals}\n')

        logging.info(f'Test results for run {run + 1}: {test_results}')
        logging.info(f'Credible Intervals: {credible_intervals}')

        # Stopping condition: If all credible intervals are narrow enough, stop early
        if all(ub - lb <= 0.01 for lb, ub in credible_intervals.values()):
            logging.info("Stopping early as all credible intervals are within threshold.")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train UNet multiple times and record results')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs per training run')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.000140459486381487, help='Learning rate')
    parser.add_argument('--scale', type=float, default=0.5, help='Image downscaling factor')
    parser.add_argument('--classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--max-repeats', type=int, default=20, help='Maximum number of training runs')
    parser.add_argument('--results-file', type=str, default='training_results.txt', help='File to save test results')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(n_channels=3, n_classes=args.classes)
    model.to(device)

    train_and_test_model(
        model=model,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        img_scale=args.scale,
        weight_decay=0.000633258434828415,
        momentum=0.999,
        gradient_clipping=2.0,
        max_repeats=args.max_repeats,
        results_file=args.results_file
    )
