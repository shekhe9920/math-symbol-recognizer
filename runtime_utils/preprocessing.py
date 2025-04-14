"""
preprocessing.py

This module defines a preprocessing pipeline for preparing handwritten math images before symbol segmentation.
It includes grayscale normalization, noise detection, denoising, thresholding, small component removal,
edge enhancement, and final binary inversion.

Key steps:
- Detects noise type (salt-and-pepper or unknown)
- Applies appropriate denoising filter
- Applies Gaussian blur, adaptive thresholding, and removes small artifacts
- Computes and overlays edge maps (Canny, Sobel, Prewitt, Roberts)
- Returns a cleaned, high-contrast binary image ready for symbol segmentation

Used by: `segment_and_predict.py` during live prediction
"""

import logging
import cv2
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter
from skimage.util import img_as_float



def load_image(path: str) -> np.ndarray:
    """
    Loads a grayscale image and converts it to float format in range [0, 1].
    """
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img_as_float(image)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Converts a float image in range [0, 1] to 8-bit unsigned integers.
    """
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def detect_noise_type(image: np.ndarray) -> str:
    """
    Heuristically detects the noise type based on pixel distribution.
    Currently detects salt-and-pepper noise.
    """
    image = np.clip(image, 0, 1)
    zeros = np.count_nonzero(image == 0)
    ones = np.count_nonzero(image == 1)

    if zeros > 0.05 * image.size or ones > 0.05 * image.size:
        print("salt_pepper")
        return "salt_pepper"
    else:
        print("unknown")
        return "unknown"


def denoise(image: np.ndarray, noise_type: str) -> np.ndarray:
    """
    Applies denoising based on detected noise type.
    - Salt & pepper: Median filter
    - Otherwise: Non-local means denoising
    """
    logging.info(f"Detected noise type: {noise_type}")
    if noise_type == "salt_pepper":
        return median_filter(image, size=3)
    else:
        return cv2.fastNlMeansDenoising(to_uint8(image), None, 10, 7, 21) / 255.0


def remove_small_components(binary: np.ndarray, min_area: int = 80) -> np.ndarray:
    """
    Removes connected components smaller than a minimum area in binary images.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            cleaned[labels == i] = 255
    return cleaned


def edge_detection_methods(image_bgr):
    """
    Applies multiple edge detection techniques and merges them.
    Returns a dictionary of individual and merged edge maps.
    """
    image_bgr = cv2.convertScaleAbs(image_bgr, alpha=1.4, beta=20)  # Boost contrast

    # Canny
    edges_canny = cv2.Canny(image_bgr, threshold1=20, threshold2=150)

    # Sobel
    sobelx = cv2.Sobel(image_bgr, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image_bgr, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Prewitt
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewittx = cv2.filter2D(image_bgr, -1, kernelx)
    prewitty = cv2.filter2D(image_bgr, -1, kernely)
    prewitt = cv2.add(prewittx, prewitty)

    # Roberts
    roberts_v = np.array([[1, 0], [0, -1]])
    roberts_h = np.array([[0, 1], [-1, 0]])
    roberts_x = cv2.filter2D(image_bgr, -1, roberts_v)
    roberts_y = cv2.filter2D(image_bgr, -1, roberts_h)
    roberts = cv2.add(roberts_x, roberts_y)

    # Merge all
    edge_stack = [edges_canny, sobel, prewitt, roberts]
    merged = np.max(np.stack(edge_stack, axis=0), axis=0)

    return {
        "Canny": edges_canny,
        "Sobel": sobel,
        "Prewitt": prewitt,
        "Roberts": roberts,
        "Merged": merged
    }


def overlay_edges_on_gray_image(gray_image, edge_map):
    """
    Overlays edge map onto grayscale image (edge lines as white).
    """
    edge_mask = cv2.threshold(edge_map, 30, 255, cv2.THRESH_BINARY)[1]
    result = gray_image.copy()
    result[edge_mask == 255] = 255
    return result


def imageprocess(image_path: str) -> np.ndarray:
    """
    Main preprocessing pipeline used before symbol segmentation.

    Steps:
    - Load grayscale image as float
    - Detect and reduce noise
    - Convert to BGR and blur
    - Adaptive thresholding (binary inversion)
    - Remove small connected components
    - Apply edge detection and overlay edges
    - Invert final result to make symbols black on white background
    """
    gray_float = load_image(image_path)
    noise_type = detect_noise_type(gray_float)
    denoised = denoise(gray_float, noise_type)

    denoised_bgr = cv2.cvtColor(to_uint8(denoised), cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(denoised_bgr, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 5
    )

    cleaned = remove_small_components(binary, min_area=80)
    edge_maps = edge_detection_methods(cleaned)
    overlay = overlay_edges_on_gray_image(cleaned, edge_maps["Merged"])

    inverted = cv2.bitwise_not(overlay)
    return inverted
