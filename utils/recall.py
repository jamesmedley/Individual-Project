import torch
from torch import Tensor


def recall(input: Tensor, target: Tensor, epsilon: float = 1e-6) -> float:
    """
    Compute the recall for binary segmentation tasks.

    Args:
        input (Tensor): Predicted binary segmentation map (N, H, W).
        target (Tensor): Ground truth binary segmentation map (N, H, W).
        epsilon (float): Small constant for numerical stability.

    Returns:
        float: Recall score.
    """
    assert input.size() == target.size(), "Input and target must have the same dimensions."

    true_positive = (input * target).sum().float()
    actual_positive = target.sum().float()

    recall_score = (true_positive + epsilon) / (actual_positive + epsilon)
    return recall_score.item()
