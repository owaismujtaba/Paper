import os
import sys
import yaml
import librosa
import logging
import numpy as np
from pathlib import Path
import scipy
from scipy import fftpack
from scipy.signal import  hilbert

from mne.filter import filter_data

hilbert3 = lambda x: hilbert(x, fftpack.next_fast_len(len(x)),axis=0)[:len(x)]

from src.logging.log import setup_logger, load_config

class EEGToFeaturesExtractor:
    def __init__(self, config_path, subject_id):
        self.subject = subject_id
        self.config = load_config(config_path)
        self._setup_config()
        log_path = os.path.join(self.log_dir, "EEG-Features-Extraction.log")
        self.logger = setup_logger('EEGToFeaturesExtractor',log_path)

        
    
    def _setup_config(self):
        self.input_dir = self.config.get("input_dir")
        self.output_dir = self.config.get("output_dir")
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.n_mels = self.config.get("n_mels", 80)
        self.window_size = int(self.sample_rate*self.config.get("window_size", 0.05))
        self.frame_shift = int(self.sample_rate*self.config.get("frame_shift", 0.01))
        self.log_dir = self.config.get("log_dir", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.eeg_sr = self.config.get('eeg_sr')
        self.model_order = self.config.get('model_order')
        self.step_size = self.config.get('step_size')

    def _load_npy(self):
        self.logger.info(f'Loading npy for Sub-{self.subject}')
        path = Path(self.input_dir, f'P{self.subject}_sEEG.npy')
        self.eeg = np.load(path)
        self.logger.info(f'loaded npy for Sub-{self.subject}')
        self.channels = np.load(Path(self.input_dir, f'P{self.subject}_channels.npy'))
    
    def _clean_eeg(self):
        self.logger.info(f'Cleaning sEEG')
        clean_data = []
        clean_channels = []
        channels = self.channels
        for i in range(channels.shape[0]):
            if '+' in channels[i][0]: #EKG/MRK/etc channels
                continue
            elif channels[i][0][0] == 'E': #Empty channels
                continue
            elif channels[i][0][:2] == 'el': #Empty channels
                continue
            elif channels[i][0][0] in ['F','C','T','O','P']: #Other channels
                continue        
            else:
                clean_channels.append(channels[i])
                clean_data.append(self.eeg[:,i])
        
        self.eeg =  np.transpose(np.array(clean_data,dtype="float64"))
        self.channels = clean_channels
        self.logger.info(f'Cleaning sEEG Finished , Shape::{self.eeg.shape}')

    def _electrode_shaft_referencing(self):
        """
        Perform electrode shaft referencing by computing the mean signal
        for each shaft and subtracting it from the corresponding channels.
        """
        self.logger.info(f'Electrode shaft referencing started for Sub-{self.subject}')

        data_esr = np.zeros_like(self.eeg)

        shafts = {}
        for i, chan in enumerate(self.channels):
            shaft_name = chan[0].rstrip('0123456789')
            shafts.setdefault(shaft_name, []).append(i)
        
        shaft_averages = {
            shaft: np.mean(self.eeg[:, indices], axis=1, keepdims=True)
            for shaft, indices in shafts.items()
        }
        
        for i, chan in enumerate(self.channels):
            shaft_name = chan[0].rstrip('0123456789')
            data_esr[:, i] = self.eeg[:, i] - shaft_averages[shaft_name].squeeze()

        self.eeg = data_esr
        self.logger.info(f'Electrode shaft referencing completed')
    
    def stack_features(self):
        self.logger.info(f'Stacking features with model_order: {self.model_order} and step_size:{self.step_size}')
        features = self.eeg_freq_band_features
        num_windows = features.shape[0]
        model_order = self.model_order
        step_size = self.step_size

        feat_stacked = []

        for f_num, i in enumerate(range(model_order * step_size, num_windows - model_order * step_size)):
            ef = features[i - model_order * step_size : i + model_order * step_size + 1 : step_size, :]
            feat_stacked.append(ef)
        feat_stacked = np.array(feat_stacked)

        self.feat_stacked = feat_stacked

    def extract_freq_band_envelope(self):
        self.logger.info(f'Extracting frequency band envlopes for sEEG')
        data = scipy.signal.detrend(self.eeg, axis=0)
        data = filter_data(data.T, self.eeg_sr, 70, 170, method="iir").T
        data = filter_data(data.T, self.eeg_sr, 102, 98, method="iir").T
        data = filter_data(data.T, self.eeg_sr, 152, 148, method="iir").T

        data = np.abs(hilbert3(data))

        frame_shift_samples = int(self.eeg_sr * self.frame_shift)
        window_size_samples = int(self.eeg_sr * self.window_size)
        num_windows = int((data.shape[0] - window_size_samples) / frame_shift_samples) + 1

        feat = []

        for win in range(num_windows):
            start = win * frame_shift_samples
            stop = start + window_size_samples
            feat.append(data[start:stop, :])

        self.eeg_freq_band_features = np.array(feat)
        self.logger.info(f'Done Extracting frequency band envelopes for sEEG')

        
        return self.eeg_freq_band_features
    
    def save_features(self):
        filename = f'P{self.subject}_eeg_features.npy'
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = Path(self.output_dir, filename)

        np.save(filepath, self.feat_stacked)
    
    def extract(self):
        self._load_npy()
        self._clean_eeg()
        self._electrode_shaft_referencing()
        self.extract_freq_band_envelope()
        self.stack_features()


def extract_eeg_features(config_path):
    for index in range(1, 31):
        if index < 10:
            index=  f'0{index}'
        extractor = EEGToFeaturesExtractor(config_path=config_path, subject_id=f'{index}')
        extractor.extract()
        extractor.save_features()