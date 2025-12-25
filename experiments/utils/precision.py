import torch
from torch import Tensor


def precision(input: Tensor, target: Tensor, epsilon: float = 1e-6) -> float:
    """
    Compute the precision for binary segmentation tasks.

    Args:
        input (Tensor): Predicted binary segmentation map (N, H, W).
        target (Tensor): Ground truth binary segmentation map (N, H, W).
        epsilon (float): Small constant for numerical stability.

    Returns:
        float: Precision score.
    """
    assert input.size() == target.size(), "Input and target must have the same dimensions."

    true_positive = (input * target).sum().float()
    predicted_positive = input.sum().float()

    precision_score = (true_positive + epsilon) / (predicted_positive + epsilon)
    return precision_score.item()
