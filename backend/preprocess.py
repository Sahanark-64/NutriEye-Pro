"""
preprocess.py - Image preprocessing utilities for NutriEye
Handles resizing, normalization, and enhancement for better predictions.
"""

import cv2
import numpy as np

IMG_SIZE = (300, 300)  # EfficientNetB3 input size


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk, apply preprocessing, and return
    a normalized numpy array ready for model inference.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at: {image_path}")

    # Convert BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Enhance contrast using CLAHE on the L channel (LAB color space)
    img = enhance_contrast(img)

    # Resize to model input size
    img = cv2.resize(img, IMG_SIZE)

    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0

    # Add batch dimension: (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)

    return img


def enhance_contrast(img_rgb: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to improve image quality under poor lighting conditions.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L (lightness) channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge back and convert to RGB
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    return enhanced


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess a raw webcam frame (numpy array in BGR format).
    Returns a normalized array ready for model inference.
    """
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = enhance_contrast(img)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img
