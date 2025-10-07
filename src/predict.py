"""High-level prediction utilities used by the Flask app and notebooks."""
from typing import Optional
import numpy as np
import logging

from .preprocessing import extract_features
from .model import load_model, predict_from_features


logger = logging.getLogger(__name__)


# The training dataset used the following 14 folders (TESS subsets):
# Order chosen to match typical os.listdir order on the training environment.
DEFAULT_LABELS = [
    'OAF_angry',
    'OAF_disgust',
    'OAF_Fear',
    'OAF_happy',
    'OAF_neutral',
    'OAF_Pleasant_surprise',
    'OAF_Sad',
    'YAF_angry',
    'YAF_disgust',
    'YAF_fear',
    'YAF_happy',
    'YAF_neutral',
    'YAF_pleasant_surprised',
    'YAF_sad',
]

# Map folder names to readable emotion labels
EMOTION_MAP = {
    'OAF_angry': 'ANGRY',
    'OAF_disgust': 'DISGUST',
    'OAF_Fear': 'FEAR',
    'OAF_happy': 'HAPPY',
    'OAF_neutral': 'NEUTRAL',
    'OAF_Pleasant_surprise': 'SURPRISED',
    'OAF_Sad': 'SAD',
    'YAF_angry': 'ANGRY',
    'YAF_disgust': 'DISGUST',
    'YAF_fear': 'FEAR',
    'YAF_happy': 'HAPPY',
    'YAF_neutral': 'NEUTRAL',
    'YAF_pleasant_surprised': 'SURPRISED',
    'YAF_sad': 'SAD',
}


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

        raw_label = labels[pred_idx] if pred_idx < len(labels) else None
        predicted_label = EMOTION_MAP.get(raw_label, 'UNKNOWN')
        confidence = float(np.max(preds))
        return {
            'label': predicted_label,
            'raw_label': raw_label,
            'confidence': confidence,
            'raw': preds.tolist(),
        }
    except Exception as e:
        logger.exception("Prediction failed for %s", audio_path)
        raise
