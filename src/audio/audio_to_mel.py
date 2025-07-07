import os
import sys
import yaml
import librosa
import logging
import numpy as np
from pathlib import Path

from src.logging.log import setup_logger, load_config

class AudioFeatureExtractor:

    def __init__(self, config_path):
        self.setup_config(config_path)


    def setup_config(self, config_path):
        self.config = load_config(config_path)
        self.input_dir = os.path.abspath(self.config.get("input_dir"))
        self.output_dir = os.path.abspath(self.config.get("output_dir"))
        self.audio_sample_rate = self.config.get("audio_sample_rate", 16000)
        self.n_mels = self.config.get("n_mels", 80)
        self.window_size = int(self.audio_sample_rate * self.config.get("window_size", 0.05))
        self.frame_shift = int(self.audio_sample_rate * self.config.get("frame_shift", 0.01))
        self.log_dir = os.path.abspath(self.config.get("log_dir", "logs"))
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, "audio-mel-feature-extraction.log")
        self.logger = setup_logger('AudioFeatureExtractor', log_path)


    def save_feature(self, subject_id):
        mel_path = os.path.join(self.output_dir, f"P{subject_id}_mel_features.npy")
        try:
            np.save(mel_path, self.mel_features)
            self.logger.info(f"Saved feature: {mel_path}")
        except Exception as e:
            self.logger.error(f"Failed to save feature to {mel_path}: {e}")


    def convert(self, subject_id):
        wav_path = Path(self.input_dir, f"P{subject_id}_audio.wav")
        

        if not os.path.exists(wav_path):
            self.logger.error(f"Numpy file for subject '{subject_id}' does not exist: {npy_path}")
            sys.exit(1)

        os.makedirs(self.output_dir, exist_ok=True)

        try:
            y, original_sr = librosa.load(wav_path, sr=self.audio_sample_rate)  
            # Optionally check shape/dtype here
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=self.sample_rate,
                n_fft=self.window_size,
                hop_length=self.frame_shift,
                n_mels=self.n_mels,
                power=2.0,
            )
            self.mel_features = mel_spec

            # mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            self.save_feature()
        except Exception as e:
            self.logger.error(f"Error processing subject '{subject_id}': {e}")


def audio_to_mel_featues():
    config_path = "configs/feature_extraction.yaml"
    extractor = AudioFeatureExtractor(config_path)
    for subject_id in range(1, 31): 
        subject_id = str(subject_id).zfill(2)
        extractor.convert(subject_id)
        extractor.save_feature()

