import os
import sys
import yaml
import librosa
import logging
import numpy as np
from pathlib import Path

from src.logging.log import setup_logger, load_config

class MelToAudioConverter:
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.input_dir = os.path.abspath(self.config.get("input_dir"))
        self.output_dir = os.path.abspath(self.config.get("output_dir"))
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.n_mels = self.config.get("n_mels", 80)
        self.n_fft = self.config.get("n_fft", 1024)
        self.hop_length = self.config.get("hop_length", 256)
        self.n_iter = self.config.get("n_iter", 60)
        self.log_dir = os.path.abspath(self.config.get("log_dir", "logs"))
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, "mel-audio-conversion.log")
        self.logger = setup_logger('MelToAudioConverter', log_path)

   

    def convert(self):
        if not os.path.exists(self.input_dir):
            self.logger.error(f"Input directory does not exist: {self.input_dir}")
            sys.exit(1)

        os.makedirs(self.output_dir, exist_ok=True)

        files = [f for f in os.listdir(self.input_dir) if f.lower().endswith("_mel.npy")]
        self.logger.info(f"Found {len(files)} mel spectrogram files to convert")

        for file in files:
            mel_path = os.path.join(self.input_dir, file)
            wav_path = os.path.join(self.output_dir, file.replace("_mel.npy", "_reconstructed.wav"))

            try:
                mel_spec = np.load(mel_path)

                # Convert mel-spectrogram to linear-frequency spectrogram using pseudo-inverse
                mel_basis = librosa.filters.mel(
                    sr=self.sample_rate,
                    n_fft=self.n_fft,
                    n_mels=self.n_mels
                )
                inv_mel_basis = np.linalg.pinv(mel_basis)
                linear_spec = np.dot(inv_mel_basis, mel_spec)

                # Reconstruct phase using Griffin-Lim
                audio = librosa.griffinlim(
                    linear_spec,
                    n_iter=self.n_iter,
                    hop_length=self.hop_length,
                    win_length=self.n_fft
                )

                librosa.output.write_wav(wav_path, audio, sr=self.sample_rate)
                self.logger.info(f"Saved reconstructed audio: {wav_path}")

            except Exception as e:
                self.logger.error(f"Error processing '{file}': {e}")
