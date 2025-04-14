"""
Organize_dataset.py

This script organizes an unsorted dataset of symbol images into a structured directory
tree. It reads images from a source folder, automatically detects the label (e.g., "0", "1", "x", "plus")
based on the filename prefix, and copies the images into their corresponding subfolders
in the destination directory.

- Deletes any existing destination dataset (can be disabled)
- Automatically creates subfolders for all valid symbol classes
- Copies images to the correct folder based on their filename prefix

Useful for preparing symbol datasets for machine learning.
"""

import os
import shutil



# path to the folder containing the unsorted symbol images
source_folder = r"C:\Users\shekh\Desktop\Skole\24.25\25\Computer Vision\camculator\bhmsds\symbols"

# path to the target dataset directory inside your Python project
destination_folder = r"C:\Users\shekh\Desktop\Skole\24.25\25\Computer Vision\Math-Solver\mathsolver\dataset\train"


# optionally delete the existing dataset directory before organizing new images
if os.path.exists(destination_folder):
    print("Deleting existing dataset...")
    shutil.rmtree(destination_folder)

# List of valid symbols and folder names
symbols = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "dot": "dot", "minus": "minus", "plus": "plus", "slash": "slash",
    "w": "w", "x": "x", "y": "y", "z": "z"
}


# Create folders for each symbol in the destination dataset directory
for folder in symbols.values():
    os.makedirs(os.path.join(destination_folder, folder), exist_ok=True)


# Iterate through source images and copy them to their respective folders
for filename in os.listdir(source_folder):
    # Extract label prefix from filename (e.g., "5-0001.png" -> "5")
    symbol_key = filename.split("-")[0]

    # Only process files with valid labels
    if symbol_key in symbols:
        src_path = os.path.join(source_folder, filename)
        dest_path = os.path.join(destination_folder, symbols[symbol_key], filename)
        shutil.copy2(src_path, dest_path)  # Copy instead of move to keep source intact

print("Dataset has been copied and organized!")
