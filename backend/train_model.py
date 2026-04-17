"""
train_model.py - Train a MobileNetV2-based food classifier for NutriEye

DATASET SETUP (do this before running):
1. Download Food-101 dataset: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
   OR use a smaller subset from Kaggle: https://www.kaggle.com/datasets/dansbecker/food-101
2. Extract so your folder looks like:
      backend/
        dataset/
          train/
            apple/  (images...)
            banana/ (images...)
            pizza/  (images...)
            ...
          val/
            apple/
            banana/
            ...

   The class folder names MUST match the food names in nutrition.csv

RUN THIS FILE ONCE to train and save the model:
   cd backend
   python train_model.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ── Configuration ──────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 20
DATASET_DIR = "dataset"          # relative to backend/
MODEL_PATH  = "food_model.h5"    # saved model output
# ───────────────────────────────────────────────────────────────────────────────


def build_model(num_classes: int) -> Model:
    """
    Build a transfer-learning model using MobileNetV2 as the base.
    MobileNetV2 is pretrained on ImageNet, so it already understands
    shapes, textures, and colors — perfect for food recognition.
    """
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,          # remove ImageNet classification head
        weights="imagenet"
    )

    # Freeze base layers first (train only the new head)
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)                          # reduce overfitting
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=predictions)
    return model, base


def get_data_generators():
    """
    Create training and validation data generators with augmentation.
    Augmentation helps the model generalize to real-world food photos.
    """
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],   # simulate different lighting
        fill_mode="nearest"
    )

    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val_data = val_gen.flow_from_directory(
        os.path.join(DATASET_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    return train_data, val_data


def train():
    if not os.path.exists(DATASET_DIR):
        print("ERROR: 'dataset/' folder not found.")
        print("Please download Food-101 and organize it as described at the top of this file.")
        return

    train_data, val_data = get_data_generators()
    num_classes = train_data.num_classes
    print(f"Found {num_classes} food classes: {list(train_data.class_indices.keys())}")

    # Save class index mapping so predict.py knows which index = which food
    import json
    class_map = {v: k for k, v in train_data.class_indices.items()}
    with open("class_indices.json", "w") as f:
        json.dump(class_map, f, indent=2)
    print("Saved class_indices.json")

    model, base = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
    ]

    print("\n── Phase 1: Training classification head ──")
    model.fit(train_data, validation_data=val_data, epochs=10, callbacks=callbacks)

    # Fine-tune: unfreeze last 30 layers of MobileNetV2
    print("\n── Phase 2: Fine-tuning MobileNetV2 layers ──")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
