import torchvision.transforms as T
from torchvision.transforms import functional as F
import random


class JointTransform:
    def __init__(self, image_transforms=None):
        self.image_transforms = image_transforms

    def __call__(self, image, mask):
        # Apply each transformation synchronously to image and mask
        if self.image_transforms:
            for t in self.image_transforms:
                if isinstance(t, T.RandomHorizontalFlip):
                    if random.random() < t.p:
                        image = F.hflip(image)
                        mask = F.hflip(mask)
                elif isinstance(t, T.RandomVerticalFlip):
                    if random.random() < t.p:
                        image = F.vflip(image)
                        mask = F.vflip(mask)
                elif isinstance(t, T.RandomRotation):
                    angle = random.uniform(t.degrees[0], t.degrees[1])  # Get random angle within specified range
                    image = F.rotate(image, angle)
                    mask = F.rotate(mask, angle)
                # Add more transformations as needed
                elif isinstance(t, T.ColorJitter):
                    image = t(image)
                # Continue for other transformations

        return image, mask
