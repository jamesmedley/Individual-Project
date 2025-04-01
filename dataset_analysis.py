import os
from PIL import Image


def find_largest_and_smallest_images(directory):
    largest_image = None
    smallest_image = None
    largest_size = (0, 0)
    smallest_size = (float('inf'), float('inf'))

    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            file_path = os.path.join(directory, filename)

            # Open the image file
            with Image.open(file_path) as img:
                width, height = img.size

                # Check if this image is the largest so far
                if (width * height) > (largest_size[0] * largest_size[1]):
                    largest_image = filename
                    largest_size = (width, height)

                # Check if this image is the smallest so far
                if (width * height) < (smallest_size[0] * smallest_size[1]):
                    smallest_image = filename
                    smallest_size = (width, height)

    return largest_image, largest_size, smallest_image, smallest_size


# Example usage
directory_path = 'data/train/imgs'  # Change this to your directory
largest, largest_size, smallest, smallest_size = find_largest_and_smallest_images(directory_path)

print(f"Largest image: {largest} with size {largest_size}")
print(f"Smallest image: {smallest} with size {smallest_size}")