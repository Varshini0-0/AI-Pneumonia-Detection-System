"""
utils.py – Model utilities for AI Pneumonia Detection System
============================================================
Handles:
  • Model loading
  • Image preprocessing (resize → normalise)
  • Grad-CAM heatmap generation
  • Demo heatmap generation (when no real model is present)
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load
from tensorflow.keras.preprocessing import image as keras_image

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)          # Input size expected by DenseNet / custom CNN
GRADCAM_LAYER = 'conv5_block16_2_conv'   # Last conv layer in DenseNet121


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model(model_path: str):
    """
    Load a Keras .h5 model from disk.
    Returns None if the file does not exist so the app can fall back to demo mode.
    """
    try:
        model = keras_load(model_path, compile=False)
        print(f"[INFO] Model loaded from '{model_path}'")
        return model
    except (OSError, IOError) as e:
        print(f"[WARN] Could not load model from '{model_path}': {e}")
        return None


# ─── Image Preprocessing ──────────────────────────────────────────────────────

def preprocess_image(img_path: str) -> np.ndarray:
    """
    Load an image from disk, resize to IMG_SIZE, convert to RGB,
    normalise pixel values to [0, 1], and add a batch dimension.

    Returns:
        np.ndarray of shape (1, 224, 224, 3)
    """
    img       = keras_image.load_img(img_path, target_size=IMG_SIZE, color_mode='rgb')
    img_array = keras_image.img_to_array(img)          # (224, 224, 3)
    img_array = img_array / 255.0                       # normalise
    img_array = np.expand_dims(img_array, axis=0)       # (1, 224, 224, 3)
    return img_array


# ─── Grad-CAM ─────────────────────────────────────────────────────────────────

def generate_gradcam(
    model,
    img_array: np.ndarray,
    original_img_path: str,
    save_path: str,
    layer_name: str = GRADCAM_LAYER,
) -> None:
    """
    Compute Grad-CAM for the given model and image, overlay the heatmap on the
    original X-ray, and save the composite image to *save_path*.

    Algorithm:
      1. Build a sub-model that outputs both the target conv-layer activations
         and the final class predictions.
      2. Record gradients of the top predicted class score w.r.t. the conv layer.
      3. Pool gradients spatially → per-channel weights → weighted sum of activations.
      4. ReLU + normalise → resize to original dimensions → apply colour map.
      5. Alpha-blend heatmap with the original image.
    """
    # ── 1. Grad-CAM model ─────────────────────────────────────────────────
    try:
        conv_layer = model.get_layer(layer_name)
    except ValueError:
        # Fall back to the last Conv2D layer in the model
        conv_layer = next(
            (l for l in reversed(model.layers) if isinstance(l, tf.keras.layers.Conv2D)),
            model.layers[-2],
        )

    grad_model = tf.keras.Model(
        inputs  = model.inputs,
        outputs = [conv_layer.output, model.output],
    )

    # ── 2. Compute gradients ──────────────────────────────────────────────
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # Use the score of the predicted class
        class_idx = tf.argmax(predictions[0])
        class_score = predictions[:, class_idx]

    grads = tape.gradient(class_score, conv_outputs)    # (1, h, w, C)

    # ── 3. Pool gradients → weighted activation sum ───────────────────────
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))    # (C,)
    conv_outputs = conv_outputs[0]                           # (h, w, C)
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)                       # (h, w)

    # ── 4. ReLU + normalise ───────────────────────────────────────────────
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # ── 5. Overlay on original image ──────────────────────────────────────
    original = cv2.imread(original_img_path)
    original = cv2.resize(original, (224, 224))

    heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap_color   = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    superimposed = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite(save_path, superimposed)


# ─── Demo Heatmap ─────────────────────────────────────────────────────────────

def generate_demo_heatmap(original_img_path: str, save_path: str) -> None:
    """
    Create a plausible-looking Grad-CAM heatmap without a real model.
    Uses Gaussian blobs in the lung region to mimic real heatmaps.
    """
    original = cv2.imread(original_img_path)
    if original is None:
        # Create a blank gray canvas if the image can't be read
        original = np.full((224, 224, 3), 128, dtype=np.uint8)
    original = cv2.resize(original, (224, 224))

    # Synthetic heatmap: two Gaussian blobs simulating bilateral lung regions
    h, w = 224, 224
    heatmap = np.zeros((h, w), dtype=np.float32)

    for cx, cy in [(80, 120), (144, 120)]:          # left & right lung centres
        for y in range(h):
            for x in range(w):
                heatmap[y, x] += np.exp(
                    -((x - cx) ** 2 + (y - cy) ** 2) / (2 * 35 ** 2)
                )

    # Normalise and apply colour map
    heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    heatmap_color   = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(original, 0.55, heatmap_color, 0.45, 0)
    cv2.imwrite(save_path, superimposed)
