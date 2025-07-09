import os
import numpy as np
from src.logging.log import load_config
from typing import Tuple

class EEGMelDataLoader:
    """
    Loads EEG and mel spectrogram features for a given subject from a single directory.
    Ensures both arrays are aligned in time.
    """
    def __init__(self, config_path: str):
        """
        Initialize the loader with the directory containing both EEG and mel features.
        Args:
            config_path (str): Path to config file with 'output_dir'.
        """
        self.config = load_config(config_path)
        self.features_dir = self.config.get("output_dir")

    def load_subject(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load EEG and mel features for a subject, align them in time.
        Args:
            subject_id (str): The subject identifier (zero-padded string).
        Returns:
            Tuple[np.ndarray, np.ndarray]: EEG features (time, channels, 1), Mel features (time, features)
        Raises:
            FileNotFoundError: If either file is missing.
            ValueError: If loaded arrays have incompatible shapes.
        """
        eeg_path = os.path.join(self.features_dir, f"P{subject_id}_eeg_features.npy")
        mel_path = os.path.join(self.features_dir, f"P{subject_id}_mel_features.npy")
        if not os.path.exists(eeg_path):
            raise FileNotFoundError(f"EEG features file not found: {eeg_path}")
        if not os.path.exists(mel_path):
            raise FileNotFoundError(f"Mel features file not found: {mel_path}")
        eeg = np.load(eeg_path)
        mel = np.load(mel_path)
        if eeg.ndim == 2:
            eeg = eeg[..., np.newaxis]  # (time, channels, 1)
        elif eeg.ndim != 3:
            raise ValueError(f"EEG array has unexpected shape: {eeg.shape}")
        mel = mel.T if mel.shape[0] != eeg.shape[0] else mel  # (time, features)
        if mel.ndim != 2:
            raise ValueError(f"Mel array has unexpected shape: {mel.shape}")
        min_len = min(eeg.shape[0], mel.shape[0])
        eeg = eeg[:min_len]
        mel = mel[:min_len]
        return eeg, mel
    




