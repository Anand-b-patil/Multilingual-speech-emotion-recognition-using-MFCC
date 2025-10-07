"""High-level prediction utilities used by the Flask app and notebooks."""
from typing import Optional
import numpy as np
import logging

from .preprocessing import extract_features
from .model import load_model, predict_from_features


logger = logging.getLogger(__name__)


DEFAULT_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']


def predict_emotion(audio_path: str, labels: Optional[list] = None) -> dict:
    labels = labels or DEFAULT_LABELS
    try:
        features = extract_features(audio_path)
        preds = predict_from_features(features)
        # If model outputs vectors, find argmax
        if preds.ndim > 1:
            pred_idx = int(np.argmax(preds, axis=1)[0])
        else:
            pred_idx = int(np.argmax(preds))

        predicted_label = labels[pred_idx] if pred_idx < len(labels) else 'unknown'
        confidence = float(np.max(preds))
        return {
            'label': predicted_label,
            'confidence': confidence,
            'raw': preds.tolist(),
        }
    except Exception as e:
        logger.exception("Prediction failed for %s", audio_path)
        raise
