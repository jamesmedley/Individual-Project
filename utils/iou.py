import torch
from torch import Tensor


def mean_iou(input: Tensor, target: Tensor, num_classes: int = 1, epsilon: float = 1e-6) -> Tensor:
    """
    Compute the IoU for binary or multiclass segmentation.

    Args:
        input (Tensor): Predicted segmentation map (N, H, W) for binary or (N, C, H, W) for multiclass.
        target (Tensor): Ground truth segmentation map (N, H, W).
        num_classes (int): Number of classes. Default is 1 for binary segmentation.
        epsilon (float): Small constant for numerical stability.

    Returns:
        Tensor: IoU per class if `num_classes > 1`, or a single IoU for binary segmentation.
    """
    assert input.dim() in (3, 4), "Input should have 3 (binary) or 4 (multiclass) dimensions."
    assert input.size()[-2:] == target.size()[-2:], "Input and target spatial dimensions must match."

    if input.dim() == 4:  # Multiclass case
        input = input.argmax(dim=1)  # Convert logits or one-hot to class predictions (N, H, W)

    if num_classes == 1:  # Binary case
        pred_mask = (input > 0.5).float()  # Threshold predictions
        target_mask = (target > 0.5).float()

        intersection = (pred_mask * target_mask).sum(dim=(-1, -2))  # Per-sample intersection
        union = (pred_mask + target_mask).clamp(0, 1).sum(dim=(-1, -2))  # Per-sample union

        iou = (intersection + epsilon) / (union + epsilon)  # Per-sample IoU
        return iou.mean()  # Return batch-average IoU for binary segmentation

    else:  # Multiclass case
        iou_per_class = torch.zeros(num_classes, device=input.device)
        for cls in range(num_classes):
            pred_mask = (input == cls)
            target_mask = (target == cls)

            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()

            if union == 0:
                iou_per_class[cls] = torch.nan
            else:
                iou_per_class[cls] = (intersection + epsilon) / (union + epsilon)

        return iou_per_class
