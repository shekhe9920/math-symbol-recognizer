"""
Count_preprocessed_images.py

This script counts the total number of images inside the preprocessed dataset directory.

Useful for verifying the dataset size after preprocessing and augmentation steps.
"""

import os

# path to preprocessed dataset directory
preprocessed_path = "../dataset/preprocessed/"
total_images = 0

# loop through each category folder and count images
for category in os.listdir(preprocessed_path):
    category_path = os.path.join(preprocessed_path, category)
    total_images += len(os.listdir(category_path))

print(f"Total number of images in the preprocessed folder: {total_images}")
