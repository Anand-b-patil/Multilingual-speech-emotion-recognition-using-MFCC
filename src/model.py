"""Model loading and inference utilities"""
from typing import Optional
import numpy as np
import tensorflow as tf
import threading


_model = None
_model_lock = threading.Lock()


def load_model(model_path: str = "models/emotion_model.keras") -> tf.keras.Model:
    """Load and cache the TensorFlow Keras model.

    Thread-safe: uses a lock to ensure single load in multi-threaded environments.
    """
    global _model
    with _model_lock:
        if _model is None:
            _model = tf.keras.models.load_model(model_path)
    return _model


def predict_from_features(features: np.ndarray, model: Optional[tf.keras.Model] = None) -> np.ndarray:
    """Run model prediction on features.

    Features expected shape: (n_features,) or (1, n_features). This function expands dims
    to (1, 1, n_features) if necessary to match models trained in the repo.
    Returns raw model output (probabilities or logits depending on model).
    """
    if model is None:
        model = load_model()

    arr = np.array(features)
    # If 1D (n_features,), expand to (1, 1, n_features) to match TimeDistributed+LSTM pattern
    if arr.ndim == 1:
        arr = arr[np.newaxis, np.newaxis, :]
    elif arr.ndim == 2:  # (time, features) -> add batch dim
        arr = arr[np.newaxis, ...]

    preds = model.predict(arr)
    return preds
