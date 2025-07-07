import os
import sys
import yaml
import numpy as np
from scipy.io.wavfile import write
import librosa
from pathlib import Path

from src.logging.log import setup_logger, load_config
import config as config

class NpyToWavConverter:
    def __init__(self, config_path):
        self.setup_config(config_path)
   
    def setup_config(self, config_path):
        self.config = load_config(config_path)
        self.input_dir = os.path.abspath(self.config.get("input_dir"))
        self.output_dir = os.path.abspath(self.config.get("output_dir"))
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.original_sr = self.config.get("original_sr", 48000)
        self.normalize_audio = self.config.get("normalize_audio", True)
        self.log_dir = os.path.abspath(self.config.get("log_dir", "logs"))
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, "npy-audio-conversion.log")
        self.logger = setup_logger('NpyToWavConverter', log_path)


    def convert(self, subject_id):      
        npy_path = Path(self.input_dir, f"P{subject_id}_audio.npy")

        self.logger.info(f"Starting conversion for subject '{subject_id}': {npy_path}")

        if not npy_path.exists():
            self.logger.error(f"Numpy file for subject '{subject_id}' does not exist: {npy_path}")
            return

        os.makedirs(self.output_dir, exist_ok=True)

        try:
            audio = np.load(npy_path)
            # Optional: check shape/dtype, normalize, resample, etc.
            if audio.ndim > 2:
                self.logger.warning(f"Skipping {npy_path}, unexpected shape {audio.shape}")
                return

            # Optional normalization
            if self.normalize_audio:
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val

            # Optional resampling
            if self.original_sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=self.original_sr, target_sr=self.sample_rate)
                self.logger.info(f"Resampled '{subject_id}' from {self.original_sr} Hz to {self.sample_rate} Hz")

            # Convert to int16 for WAV
            if audio.dtype != np.int16:
                audio = (audio * 32767).astype(np.int16)

            self.save_audio(audio, subject_id)
        except Exception as e:
            self.logger.error(f"Error converting '{subject_id}': {e}")

    def save_audio(self, audio, subject_id):
        wav_path = os.path.join(self.output_dir, f"P{subject_id}_audio.wav")
        self.logger.info(f"Saving WAV file to: {wav_path}")
        try:
            write(wav_path, self.sample_rate, audio)
            self.logger.info(f"Saved WAV: {wav_path}")
        except Exception as e:
            self.logger.error(f"Failed to save WAV to {wav_path}: {e}")


def npy_to_wav_converter():
    config_path = Path(config.CUR_DIR, 'configs/npy_to_wav.yaml')
    for index in range(1, 31):
        subject =  str(index).zfill(2)
        converter = NpyToWavConverter(config_path=config_path)
