import os
import sys
import yaml
import librosa
import logging
import numpy as np
from pathlib import Path

from src.logging.log import setup_logger, load_config

class AudioFeatureExtractor:
    """
    Extracts mel spectrogram features from audio files for a set of subjects.
    """

    def __init__(self, config_path: str):
        """
        Initialize the AudioFeatureExtractor with configuration.
        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.setup_config(config_path)


    def setup_config(self, config_path: str):
        """
        Load configuration and set up directories and parameters.
        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config = load_config(config_path)
        self.input_dir = os.path.abspath(self.config.get("input_dir"))
        self.output_dir = os.path.abspath(self.config.get("output_dir"))
        self.audio_sample_rate = self.config.get("audio_sample_rate", 16000)
        self.n_mels = self.config.get("n_mels", 80)
        self.window_size = int(self.audio_sample_rate * self.config.get("window_size", 0.05))
        self.frame_shift = int(self.audio_sample_rate * self.config.get("frame_shift", 0.01))
        self.log_dir = os.path.abspath(self.config.get("log_dir", "logs"))
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, "feature-extraction.log")
        self.logger = setup_logger('AudioFeatureExtractor', log_path)


    def save_feature(self, subject_id: str):
        """
        Save the extracted mel features for a subject to a .npy file.
        Args:
            subject_id (str): The subject identifier (zero-padded string).
        """
        mel_path = os.path.join(self.output_dir, f"P{subject_id}_mel_features.npy")
        try:
            np.save(mel_path, self.mel_features)
            self.logger.info(f"Saved feature: {mel_path}")
        except Exception as e:
            self.logger.error(f"Failed to save feature to {mel_path}: {e}")


    def load_wav_file(self, wav_path: Path):
        """
        Load a WAV file and return the audio array and sample rate.
        Args:
            wav_path (Path): Path to the WAV file.
        Returns:
            Tuple[np.ndarray, int]: Audio array and sample rate, or (None, None) if failed.
        """
        self.logger.info(f"Loading WAV file: {wav_path}")
        try:
            y, sr = librosa.load(wav_path, sr=self.audio_sample_rate)
            return y, sr
        except Exception as e:
            self.logger.error(f"Failed to load WAV file {wav_path}: {e}")
            return None, None


    def convert(self, subject_id: str):
        """
        Convert a subject's WAV file to a mel spectrogram and store in the instance.
        Args:
            subject_id (str): The subject identifier (zero-padded string).
        """
        wav_path = Path(self.input_dir, f"P{subject_id}_audio.wav")
        self.logger.info(f"Starting conversion for subject '{subject_id}': {wav_path}")
        try:
            y, original_sr = self.load_wav_file(wav_path)
            if y is None:
                return
            # Optionally check shape/dtype here
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=self.audio_sample_rate,
                n_fft=self.window_size,
                hop_length=self.frame_shift,
                n_mels=self.n_mels,
                power=2.0,
            )
            self.mel_features = mel_spec

            # mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        except Exception as e:
            self.logger.error(f"Error processing subject '{subject_id}': {e}")


def audio_to_mel_features():
    """
    Batch process all subjects to extract mel features from their audio files.
    """
    config_path = "configs/feature_extraction.yaml"
    extractor = AudioFeatureExtractor(config_path)
    for subject_id in range(1, 31): 
        subject_id_str = str(subject_id).zfill(2)
        extractor.convert(subject_id_str)
        extractor.save_feature(subject_id_str)

