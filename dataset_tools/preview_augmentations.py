"""
preview_augmentations.py

This script previews how various data augmentations affect a single preprocessed image.
It simulates real-world distortions like noise, blur, and rotation to test the visual outcome
before applying the transformations to the entire dataset.

Steps:
- Loads and preprocesses a 28x28 grayscale image (resize, denoise, contrast, binarize)
- Applies augmentations: Gaussian noise, blur, random rotation, and brightness/contrast boost
- Displays the results side-by-side using matplotlib

Purpose: Visual debugging and fine-tuning of your augmentation pipeline for model robustness.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


test_image_path = "../dataset/train/1/1-0001.png"  # image to test




# Function to preprocess the image (resizing, denoising, contrast enhancement, binarization)
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Resize the image to 28x28 pixels (same as model input size)
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    # Apply bilateral filter to reduce noise while keeping edges sharp
    image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Apply CLAHE to boost local contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    image = clahe.apply(image)

    # Apply Otsu thresholding to binarize the image (turn into black & white)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return image




# Function to create several augmentations of the preprocessed image
def augment_image(image):
    augmentations = []

    # Original image (no changes)
    augmentations.append(("Original", image))

    # Add mild Gaussian noise to simulate dirty or low-quality input
    noisy = image.copy().astype(np.float32)
    noise = np.random.normal(0, 15, noisy.shape).astype(np.float32)
    noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)
    augmentations.append(("Gaussian Noise", noisy))

    # Apply slight Gaussian blur to simulate out-of-focus images
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    augmentations.append(("Blurred", blurred))

    # Rotate the image randomly between -10 and +10 degrees
    rows, cols = image.shape
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows), borderValue=255)
    augmentations.append(("Rotated", rotated))

    # Increase brightness and contrast slightly
    bright_contrast = cv2.convertScaleAbs(image, alpha=1.1, beta=20)
    augmentations.append(("Brightness/Contrast", bright_contrast))

    return augmentations





image = preprocess_image(test_image_path)
augmented_images = augment_image(image)

plt.figure(figsize=(10, 2))

for i, (title, aug) in enumerate(augmented_images):
    plt.subplot(1, len(augmented_images), i+1)
    plt.imshow(aug, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()
