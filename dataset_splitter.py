import os
import shutil
import random
from pathlib import Path

# Paths to the original data
DATA_DIR = Path('./data')
IMG_DIR = DATA_DIR / 'imgs'
MASK_DIR = DATA_DIR / 'masks'

# Paths for train, validation, and test folders
TRAIN_IMG_DIR = DATA_DIR / 'train/imgs'
TRAIN_MASK_DIR = DATA_DIR / 'train/masks'
VAL_IMG_DIR = DATA_DIR / 'val/imgs'
VAL_MASK_DIR = DATA_DIR / 'val/masks'
TEST_IMG_DIR = DATA_DIR / 'test/imgs'
TEST_MASK_DIR = DATA_DIR / 'test/masks'

# Ensure directories exist
for folder in [TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, TEST_IMG_DIR, TEST_MASK_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Get list of all images and masks
img_files = sorted(IMG_DIR.glob('*'))  # Assumes images have unique filenames
mask_files = sorted(MASK_DIR.glob('*'))  # Assumes masks align 1-to-1 with images

assert len(img_files) == len(mask_files), "Mismatch between number of images and masks."

# Shuffle the data
data = list(zip(img_files, mask_files))
random.seed(42)  # Ensure reproducibility
random.shuffle(data)

# Split data into train, val, and test sets (80:10:10)
n_total = len(data)
n_train = int(n_total * 0.8)
n_val = int(n_total * 0.1)
n_test = n_total - n_train - n_val

train_data = data[:n_train]
val_data = data[n_train:n_train + n_val]
test_data = data[n_train + n_val:]

# Function to move files
def move_files(data_split, img_dest, mask_dest):
    for img_path, mask_path in data_split:
        shutil.copy(img_path, img_dest / img_path.name)
        shutil.copy(mask_path, mask_dest / mask_path.name)

# Move files into respective folders
move_files(train_data, TRAIN_IMG_DIR, TRAIN_MASK_DIR)
move_files(val_data, VAL_IMG_DIR, VAL_MASK_DIR)
move_files(test_data, TEST_IMG_DIR, TEST_MASK_DIR)

print(f"Data successfully split and moved:")
print(f"Training set: {len(train_data)} images and masks")
print(f"Validation set: {len(val_data)} images and masks")
print(f"Test set: {len(test_data)} images and masks")
