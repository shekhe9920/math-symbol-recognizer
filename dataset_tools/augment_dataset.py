"""
This script preprocesses and augments all images in the training dataset.

Each image is resized, denoised, enhanced, and binarized before generating five augmented variants:
- original
- with Gaussian noise
- with Gaussian blur
- with random rotation
- with brightness/contrast adjustment

The output is saved in a new folder structure under '../dataset/preprocessed'.
"""

import os
import cv2
import numpy as np
import shutil
import random


dataset_path = "../dataset/train"  # input
output_path = "../dataset/preprocessed"  # output




# To reprocess the dataset
if os.path.exists(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path, exist_ok=True)




# Function to preprocess a single image before augmentation
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize to 28x28 pixels
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    # Denoise using bilateral filter to preserve edges
    image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Apply CLAHE to enhance contrast (adaptive histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    image = clahe.apply(image)

    # Apply Otsu's thresholding to binarize the image
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return image




# Function to create augmented versions of the preprocessed image
def augment_image(image):
    augmentations = []

    # Original image (preprocessed)
    augmentations.append(image)

    # Add mild Gaussian noise
    noisy = image.astype(np.float32)
    noise = np.random.normal(0, 5, noisy.shape).astype(np.float32)
    noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)
    augmentations.append(noisy)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    augmentations.append(blurred)

    # Apply random rotation between -10 and +10 degrees
    rows, cols = image.shape
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows), borderValue=255)
    augmentations.append(rotated)

    # Slight brightness and contrast adjustment
    bright_contrast = cv2.convertScaleAbs(image, alpha=1.1, beta=20)
    augmentations.append(bright_contrast)

    return augmentations




# Process all images in the dataset and apply augmentations
for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    output_category_path = os.path.join(output_path, category)
    os.makedirs(output_category_path, exist_ok=True)

    for filename in os.listdir(category_path):
        image_path = os.path.join(category_path, filename)
        processed_image = preprocess_image(image_path)
        augmented_images = augment_image(processed_image)

        # Save all augmented versions to the output directory
        for i, aug_img in enumerate(augmented_images):
            output_file = os.path.join(output_category_path, f"{filename[:-4]}_aug{i}.png")
            cv2.imwrite(output_file, aug_img)

print("Dataset has been preprocessed and matches the preview results:", output_path)
