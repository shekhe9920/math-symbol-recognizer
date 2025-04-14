"""
Plot_training_history.py

This script loads the saved training history (stored as a pickle file) and visualizes
the training and validation loss and accuracy curves over epochs.

Useful for analyzing model performance, overfitting, and generalization after training.
"""

import pickle
import matplotlib.pyplot as plt


# Load the training history from pickle file
history_path = "../models/training_history.pkl"
with open(history_path, "rb") as f:
    history = pickle.load(f)



# Plot training and validation loss
plt.plot(history["loss"], label="Training Loss")
plt.plot(history.get("val_loss", []), label="Validation Loss")  # Only plot if available
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training History - Loss")
plt.show()



# Plot training and validation accuracy (if available)
if "accuracy" in history:
    plt.plot(history["accuracy"], label="Training Accuracy")
    plt.plot(history.get("val_accuracy", []), label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training History - Accuracy")
    plt.show()
