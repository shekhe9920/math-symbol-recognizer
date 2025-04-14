"""
train.py

Trains a Convolutional Neural Network (CNN) to recognize handwritten mathematical symbols
(digits, operators, variables) using preprocessed and augmented training data.

The model architecture includes multiple convolutional layers, batch normalization,
and dropout for regularization. It uses early stopping and learning rate scheduling
to avoid overfitting and improve convergence.

Outputs:
- Trained model saved as 'digit_classifier.h5'
- Training history saved as 'training_history.pkl'
"""

import pickle
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# paths to dataset and model save location
dataset_path = "../dataset/preprocessed"
model_save_path = "../models/digit_classifier.h5"

# Model parameters
img_size = (28, 28)
batch_size = 32
epochs = 100
num_classes = 19  # total number of classes (digits + operators + variables)



# data augmentation for the training set (applies random transformations)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,           # normalize pixel values to [0,1]
    rotation_range=15,             # random rotation up to 15 degrees
    width_shift_range=0.15,        # random horizontal shift
    height_shift_range=0.15,       # random vertical shift
    zoom_range=0.2,                # random zoom
    shear_range=0.2,               # random shearing
    fill_mode="nearest",           # filling strategy for empty pixels after transformation
    validation_split=0.2           # split dataset: 80% training / 20% validation
)



# Loads training set
train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)



# Loads validation set
val_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)




# CNN architecture with multiple convolutional layers and dropout for regularization
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    layers.BatchNormalization(),

    layers.Conv2D(256, (3,3), activation="relu", padding="same"),
    layers.BatchNormalization(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),

    layers.Dropout(0.4),  # Slightly increased dropout rate for better generalization

    layers.Dense(num_classes, activation="softmax")
])




# callbacks for early stopping and adaptive learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)



# compiles the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])



# training the model with validation and callbacks
history = model.fit(train_data,
                    epochs=epochs,
                    validation_data=val_data,
                    callbacks=[early_stopping, reduce_lr])



# Saves training history (for later plotting or analysis)
history_path = "../models/training_history.pkl"
with open(history_path, "wb") as f:
    pickle.dump(history.history, f)
print(f"Training history saved at {history_path}")



# Saves the trained model
model.save(model_save_path)
print(f"Model saved at {model_save_path}")
