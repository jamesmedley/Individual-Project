import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff
from utils.iou import mean_iou
from utils.precision import precision
from utils.recall import recall


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)

    # Initialize metrics
    dice_score = 0
    total_iou = 0
    total_precision = 0
    total_recall = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # Predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:  # Binary segmentation
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                mask_true = mask_true.unsqueeze(1)  # Add channel dimension
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)

                # Compute IoU, precision, and recall
                total_iou += mean_iou(mask_pred.squeeze(1), mask_true.squeeze(1), num_classes=1)
                total_precision += precision(mask_pred, mask_true)
                total_recall += recall(mask_pred, mask_true)

            else:  # Multiclass segmentation
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

                # Compute IoU, precision, and recall
                class_iou = mean_iou(mask_pred.argmax(dim=1), mask_true.argmax(dim=1), num_classes=net.n_classes)
                total_iou += class_iou.mean()  # Average IoU across classes
                total_precision += precision(mask_pred, mask_true)
                total_recall += recall(mask_pred, mask_true)

    # Compute average metrics
    dice_score /= max(num_val_batches, 1)
    total_iou /= max(num_val_batches, 1)
    total_precision /= max(num_val_batches, 1)
    total_recall /= max(num_val_batches, 1)

    net.train()
    return {
        'dice_score': dice_score,
        'mIoU': total_iou,
        'precision': total_precision,
        'recall': total_recall
    }