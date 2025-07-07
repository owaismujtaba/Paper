import os
import sys
import yaml
import librosa
import logging
import numpy as np
from pathlib import Path

from src.logging.log import setup_logger, load_config

class AudioToMelConverter:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.input_dir = os.path.abspath(self.config.get("input_dir"))
        self.output_dir = os.path.abspath(self.config.get("output_dir"))
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.n_mels = self.config.get("n_mels", 80)
        self.window_size = int(self.sample_rate*self.config.get("window_size", 0.05))
        self.frame_shift = int(self.sample_rate*self.config.get("frame_shift", 0.01))
        self.log_dir = os.path.abspath(self.config.get("log_dir", "logs"))
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, "audio-mel-conversion.log")
        self.logger = setup_logger('AudioToMelConverter',log_path)
    

    def convert(self):
        if not os.path.exists(self.input_dir):
            self.logger.error(f"Input directory does not exist: {self.input_dir}")
            sys.exit(1)

        os.makedirs(self.output_dir, exist_ok=True)

        files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(".wav")]
        self.logger.info(f"Found {len(files)} .wav files to convert")

        for file in files:
            wav_path = os.path.join(self.input_dir, file)
            mel_path = os.path.join(self.output_dir, file.replace(".wav", "_mel.npy"))

            try:
                y, original_sr = librosa.load(wav_path, sr=self.sample_rate)  
                mel_spec = librosa.feature.melspectrogram(
                    y=y,
                    sr=self.sample_rate,
                    n_fft=self.window_size,
                    hop_length=self.frame_shift,
                    n_mels=self.n_mels,
                    power=2.0,
                )
                #mel_db = librosa.power_to_db(mel_spec, ref=np.max)

                np.save(mel_path, mel_spec)
                self.logger.info(f"Saved mel spectrogram: {mel_path}")

            except Exception as e:
                self.logger.error(f"Error processing '{file}': {e}")
