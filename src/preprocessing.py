"""Audio preprocessing utilities"""
from typing import Tuple
import numpy as np
import librosa


def extract_features(file_path: str, sr: int = 22050, duration: float = None, offset: float = 0.0) -> np.ndarray:
    """Load an audio file and extract MFCC features (mean over time).

    Returns a 1D numpy array of length n_mfcc (default 13).
    """
    audio, sr = librosa.load(file_path, sr=sr, duration=duration, offset=offset)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    # Transpose to (time, features) and take mean over time axis
    features = np.mean(mfccs.T, axis=0)
    return features


def load_waveform(file_path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """Return waveform and sample rate"""
    audio, sr = librosa.load(file_path, sr=sr)
    return audio, sr
