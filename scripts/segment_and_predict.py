"""
segment_and_predict.py

Performs full math expression recognition:
- Loads the trained model
- Preprocesses image
- Segments symbols
- Predicts characters
- Displays bounding boxes and recognized expression
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === Load model and class mapping ===
model = load_model("../models/digit_classifier.h5")
class_mapping = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: ".", 11: "=", 12: "-", 13: "+", 14: "/", 15: "w", 16: "x", 17: "y", 18: "z"
}


def vertical_splits(roi, x0, y0, w, h):
    """Attempts to split wide bounding boxes into individual symbol boxes."""
    profile = np.sum(roi, axis=0)
    min_val, max_val = np.min(profile), np.max(profile)
    threshold = min_val + 0.3 * (max_val - min_val)

    cuts = [i for i in range(1, len(profile) - 1)
            if profile[i - 1] > threshold and profile[i] <= threshold and profile[i + 1] > threshold]

    filtered = []
    last = -999
    for c in cuts:
        if c - last > 10:
            filtered.append(c)
            last = c

    boxes = []
    prev = 0
    for cut in filtered:
        if cut - prev >= 8:
            boxes.append((x0 + prev, y0, cut - prev, h))
        prev = cut
    if w - prev >= 8:
        boxes.append((x0 + prev, y0, w - prev, h))

    return boxes if boxes else [(x0, y0, w, h)]



def predict_symbols(binary):
    """Detects and predicts characters from a binarized image."""
    contours, _ = cv2.findContours(255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, symbols = [], []
    img_h, img_w = binary.shape

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # Skip noise and long lines
        aspect_ratio = w / h if h != 0 else 999
        if aspect_ratio > 8 or aspect_ratio < 0.125 or w > 0.9 * img_w or h > 0.9 * img_h:
            continue

        # Split wide symbols if needed
        split_needed = w > 1.3 * h
        roi = binary[y:y + h, x:x + w]
        candidates = vertical_splits(roi, x, y, w, h) if split_needed else [(x, y, w, h)]

        for sx, sy, sw, sh in candidates:
            roi_cropped = binary[sy:sy + sh, sx:sx + sw]
            pad = int(0.25 * max(sw, sh))
            padded = np.ones((sh + 2 * pad, sw + 2 * pad), dtype=np.uint8) * 255
            padded[pad:pad + sh, pad:pad + sw] = roi_cropped

            resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)
            input_img = resized.astype("float32") / 255.0
            input_img = input_img.reshape(1, 28, 28, 1)
            pred = model.predict(input_img, verbose=0)
            pred_idx = np.argmax(pred)
            symbol = class_mapping.get(pred_idx, "?")

            if symbol == ".":
                symbol = "*"  # Treat dots as multiplication

            symbols.append(symbol)
            boxes.append((sx, sy, sw, sh))

    # Sort symbols left to right
    sorted_data = sorted(zip(boxes, symbols), key=lambda x: x[0][0])
    boxes, symbols = zip(*sorted_data) if sorted_data else ([], [])

    # Merge pairs of '-' into '='
    merged_boxes, merged_symbols = [], []
    skip_next = False
    for i in range(len(symbols)):
        if skip_next:
            skip_next = False
            continue
        if symbols[i] == "-" and i + 1 < len(symbols) and symbols[i + 1] == "-":
            x1, y1, w1, h1 = boxes[i]
            x2, y2, w2, h2 = boxes[i + 1]
            dx, dy, dh = abs(x1 - x2), abs(y1 - y2), abs(h1 - h2)
            if dx < 15 and dy > 4 and dh < 25:
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                merged_boxes.append((new_x, new_y, new_w, new_h))
                merged_symbols.append("=")
                skip_next = True
                continue
        merged_boxes.append(boxes[i])
        merged_symbols.append(symbols[i])

    return "".join(merged_symbols), merged_boxes, merged_symbols



def draw_boxes(img, boxes, symbols):
    """Draws bounding boxes and predicted symbols on the image."""
    output = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(boxes):
        label = symbols[i]
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(output, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    return output
