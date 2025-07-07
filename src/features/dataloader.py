import os
import numpy as np



class EEGMelDataLoader:
    """
    Loads EEG and mel spectrogram features for a given subject from a single directory.
    """
    def __init__(self, config_path: str):
        """
        Initialize the loader with the directory containing both EEG and mel features.
        Args:
            features_dir (str): Directory containing EEG and mel feature .npy files.
        """
        self.config = load_config(config_path)
        self.features_dir = self.config.get("output_dir")

    def load_subject(self, subject_id: str):
        """
        Load EEG and mel features for a subject.
        Args:
            subject_id (str): The subject identifier (zero-padded string).
        Returns:
            Tuple[np.ndarray, np.ndarray]: EEG features, Mel features
        Raises:
            FileNotFoundError: If either file is missing.
        """
        eeg_path = os.path.join(self.features_dir, f"P{subject_id}_eeg_features.npy")
        mel_path = os.path.join(self.features_dir, f"P{subject_id}_mel_features.npy")
        if not os.path.exists(eeg_path):
            raise FileNotFoundError(f"EEG features file not found: {eeg_path}")
        if not os.path.exists(mel_path):
            raise FileNotFoundError(f"Mel features file not found: {mel_path}")
        eeg = np.load(eeg_path)
        mel = np.load(mel_path)
        mel = mel.T
        
        min_len = min(eeg.shape[0], mel.shape[0])   
        eeg = eeg[:min_len, :]
        mel = mel[:min_len, :]
        return eeg, mel
    




