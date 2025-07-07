import os
import sys
import yaml
import numpy as np
from scipy.io.wavfile import write
import librosa
from pathlib import Path

from src.logging.log import setup_logger, load_config

class NpyToWavConverter:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.input_dir = os.path.abspath(self.config.get("input_dir"))
        self.output_dir = os.path.abspath(self.config.get("output_dir"))
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.original_sr = self.config.get("original_sr", 48000)
        self.normalize_audio = self.config.get("normalize_audio", True)
        self.log_dir = os.path.abspath(self.config.get("log_dir", "logs"))
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, "npy-audio-conversion.log")
        self.logger = setup_logger('NpyToWavConverter',log_path)
   

    def convert(self):
        if not os.path.exists(self.input_dir):
            self.logger.error(f"Input directory does not exist: {self.input_dir}")
            sys.exit(1)

        os.makedirs(self.output_dir, exist_ok=True)

        files = [f for f in os.listdir(self.input_dir) if f.endswith("audio.npy")]
        self.logger.info(f"Found {len(files)} audio files to convert")

        for file in files:
            npy_path = os.path.join(self.input_dir, file)
            wav_path = os.path.join(self.output_dir, file.replace(".npy", ".wav"))

            try:
                audio = np.load(npy_path)

                if audio.ndim > 2:
                    self.logger.warning(f"Skipping {npy_path}, unexpected shape {audio.shape}")
                    continue

                if self.normalize_audio:
                    max_val = np.max(np.abs(audio))
                    if max_val > 0:
                        audio = audio / max_val

                
                if self.original_sr != self.sample_rate:
                    audio = librosa.resample(audio, orig_sr=self.original_sr, target_sr=self.sample_rate)
                    self.logger.info(f"Resampled '{file}' from {self.original_sr} Hz to {self.sample_rate} Hz")
                if audio.dtype != np.int16:
                    audio = (audio * 32767).astype(np.int16)
                print('sample rate', self.sample_rate, wav_path)
                write(wav_path, self.sample_rate, audio)
                self.logger.info(f"Converted '{file}' to WAV")

            except Exception as e:
                self.logger.error(f"Error converting '{file}': {e}")
