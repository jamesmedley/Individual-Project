import argparse
import logging
import os
import torch
from evaluate import evaluate
from unet import UNet
from unet import ScatUNet
from pathlib import Path
from utils.data_loading import BasicDataset
from torch.utils.data import DataLoader

test_img_dir = Path('./data/test/imgs/')
test_mask_dir = Path('./data/test/masks/')



def test_model(
    model,
    device,
    img_scale: float = 0.5,
    amp: bool = False
):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    test_set = BasicDataset(test_img_dir, test_mask_dir, img_scale)
    loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)
    test_score = evaluate(model, test_loader, device, amp)
    logging.info('Learnable parameters: {}'.format(pytorch_total_params))
    logging.info('Mean Dice score: {}'.format(test_score["dice_score"]))
    logging.info('mIoU: {}'.format(test_score["mIoU"]))
    logging.info('Precision: {}'.format(test_score["precision"]))
    logging.info('Recall: {}'.format(test_score["recall"]))

    return test_score

def get_args():
    parser = argparse.ArgumentParser(description='Test a model on images and target masks')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = ScatUNet(n_channels=3, n_classes=1, bilinear=args.bilinear, J=1, L=16, input_shape=(128, 128))
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)

        del state_dict['mask_values']

        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        test_model(
            model=model,
            device=device,
            img_scale=args.scale,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        test_model(
            model=model,
            device=device,
            img_scale=args.scale,
            amp=args.amp
        )