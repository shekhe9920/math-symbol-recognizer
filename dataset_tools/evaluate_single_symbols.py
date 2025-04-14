"""
evaluate_single_symbols.py

This script evaluates a trained symbol classification model on a folder of single-symbol test images.
It performs the following steps for each image:
- Applies preprocessing (grayscale, denoising, thresholding, resizing)
- Predicts the symbol class using a pre-trained model
- Applies a confidence threshold (default 80%) to flag uncertain predictions
- Prints predicted results and overall accuracy
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras



# loads the trained model
model = keras.models.load_model("../models/digit_classifier.h5")
print("Number of classes the model has learned:", model.output_shape[-1])



# folder containing the test images
test_folder = "../test-images/single-symbols"
test_results = {}


threshold = 0.80  # confidence threshold for predictions



for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)

    # image preprocessing using OpenCV
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converts to grayscale
    img = cv2.fastNlMeansDenoising(img, None, 13, 7, 21)  # denoising

    block_size = max(img.shape) // 7
    block_size = block_size + 1 if block_size % 2 == 0 else block_size
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, block_size, 5)

    # resizing and pads the image to 28x28 pixels
    h, w = img.shape
    if h >= w:
        new_w = int(28 * w / h)
        img = cv2.resize(img, (new_w, 28))
        rest = 28 - new_w
        left = rest // 2
        right = rest - left
        img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=[255])
    else:
        new_h = int(28 * h / w)
        img = cv2.resize(img, (28, new_h))
        rest = 28 - new_h
        top = rest // 2
        bottom = rest - top
        img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=[255])

    # normalize the image and prepare it for model input
    img_array = img.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # shape: [1, 28, 28, 1]

    # runs prediction:
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = max(prediction[0])

    # confidence threshold
    if confidence < threshold:
        label = f"UNSURE (Guess: {predicted_class})"
    else:
        label = str(predicted_class)

    # store label, confidence, and raw predicted class
    test_results[img_name] = (label, confidence, predicted_class)



# results for each test image
print("\nTest images:")
for img_name, (label, confidence, _) in test_results.items():
    print(f"{img_name}: Predicted class -> {label} (confidence: {confidence:.2f})")



# accuracy calculation
correct = 0
total = len(test_results)
unsure_count = 0

for img_name, (label, confidence, predicted_class) in test_results.items():
    true_label = int(img_name.split("-")[0])
    if "UNSURE" in label:
        unsure_count += 1
    if predicted_class == true_label:
        correct += 1

accuracy = (correct / total) * 100
print(f"\nAccuracy: {correct}/{total} correct ({accuracy:.2f}%)")
print(f"Number of UNSURE images: {unsure_count}/{total}")
