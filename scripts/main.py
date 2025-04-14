"""
main_predictor.py

This script loads one or more test images, applies the preprocessing pipeline,
segments symbols, predicts them using a trained model, and visualizes the result.

It supports:
- single image testing
- batch testing
- previewing the latest image

Modules used:
- `imageprocess()` from runtime_utils.preprocessing
- `predict_symbols()` and `draw_boxes()` from segment_and_predict
"""

from segment_and_predict import predict_symbols, draw_boxes
from runtime_utils.preprocessing import imageprocess
import cv2
import glob
import os
import matplotlib.pyplot as plt


def run_on_images(image_paths):
    for path in image_paths:
        print(f"\nProcessing: {path}")
        binary = imageprocess(path)
        if len(binary.shape) == 3:
            binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)

        expr, boxes, symbols = predict_symbols(binary)
        visual = draw_boxes(binary, boxes, symbols)
        original = cv2.imread(path)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(visual, cv2.COLOR_BGR2RGB))
        plt.title(f"Recognized: {expr}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # === Testmodus ===
    TEST_MODE = "single"  # "single", "latest", "all"
    custom_image_path = "../test-images/equations/img_14.png"

    if TEST_MODE == "all":
        images = sorted(glob.glob("../test-images/equations/*.png"), key=os.path.getmtime)
    elif TEST_MODE == "single":
        images = [custom_image_path]
    else:
        images = sorted(glob.glob("../test-images/equations/*.png"), key=os.path.getmtime)[-1:]

    if not images:
        print("No images found.")
    else:
        run_on_images(images)
