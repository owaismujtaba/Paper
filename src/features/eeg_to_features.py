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
import pdb
from mne.filter import filter_data

hilbert3 = lambda x: hilbert(x, fftpack.next_fast_len(len(x)),axis=0)[:len(x)]

from src.logging.log import setup_logger, load_config

class EEGFeaturesExtractor:
    """
    A class used to extract features from EEG data.

    Attributes:
    ----------
    config : dict
        Configuration dictionary containing parameters for feature extraction.
    input_dir : str
        Directory path where input EEG data is stored.
    output_dir : str
        Directory path where output features will be saved.
    sample_rate : int
        Sample rate of the EEG data.
    window_size : int
        Size of the window for feature extraction.
    frame_shift : int
        Frame shift for feature extraction.
    log_dir : str
        Directory path where log files will be saved.
    logger : logging.Logger
        Logger instance for logging messages.
    model_order : int
        Model order for feature stacking.
    step_size : int
        Step size for feature stacking.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initializes the EEGFeaturesExtractor instance.

        Parameters:
        ----------
        config_path : str
            Path to the configuration file.
        """
        self._setup_config(config_path)

    def _setup_config(self, config_path: str) -> None:
        """
        Sets up the configuration for feature extraction.

        Parameters:
        ----------
        config_path : str
            Path to the configuration file.
        """
        self.config = load_config(config_path)
        self.input_dir = self.config.get("eeg_input_dir")
        self.output_dir = self.config.get("output_dir")
        self.sample_rate = self.config.get("eeg_sample_rate", 1024)
        self.window_size = int(self.sample_rate * self.config.get("window_size"))
        self.frame_shift = int(self.sample_rate * self.config.get("frame_shift"))
        self.log_dir = self.config.get("log_dir", "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        log_path = os.path.join(self.log_dir, "feature-extraction.log")
        self.logger = setup_logger('EEGFeaturesExtractor', log_path)
        self.model_order = self.config.get('model_order')
        self.step_size = self.config.get('step_size')

    def _load_eeg(self, subject_id: str) -> None:
        """
        Loads the EEG data for a given subject.

        Parameters:
        ----------
        subject_id : str
            ID of the subject.
        """
        self.logger.info(f'Loading npy for Sub-{subject_id}')
        path = Path(self.input_dir, f'P{subject_id}_sEEG.npy')
        self.eeg = np.load(path)
        self.logger.info(f'loaded npy for Sub-{subject_id}')
        self.channels = np.load(Path(self.input_dir, f'P{subject_id}_channels.npy'))

    def _clean_eeg(self) -> None:
        """
        Cleans the EEG data by removing unwanted channels.
        """
        self.logger.info(f'Cleaning sEEG')
        clean_data = []
        clean_channels = []
        channels = self.channels
        for i in range(channels.shape[0]):
            if '+' in channels[i][0]:  # EKG/MRK/etc channels
                continue
            elif channels[i][0][0] == 'E':  # Empty channels
                continue
            elif channels[i][0][:2] == 'el':  # Empty channels
                continue
            elif channels[i][0][0] in ['F', 'C', 'T', 'O', 'P']:  # Other channels
                continue
            else:
                clean_channels.append(channels[i])
                clean_data.append(self.eeg[:, i])

        self.eeg = np.transpose(np.array(clean_data, dtype="float64"))
        self.channels = clean_channels
        self.logger.info(f'Cleaning sEEG Finished , Shape::{self.eeg.shape}')

    def _electrode_shaft_referencing(self) -> None:
        """
        Performs electrode shaft referencing by computing the mean signal
        for each shaft and subtracting it from the corresponding channels.
        """
        self.logger.info(f'Electrode shaft referencing started')

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

    def stack_features(self) -> None:
        """
        Stacks features with model_order and step_size.
        """
        self.logger.info(f'Stacking features with model_order: {self.model_order} and step_size:{self.step_size}')
        features = self.eeg_freq_band_features
        num_windows = features.shape[0]
        model_order = self.model_order
        step_size = self.step_size

        feat_stacked = []

        for f_num, i in enumerate(range(model_order * step_size, num_windows - model_order * step_size)):
            ef = features[i - model_order * step_size: i + model_order * step_size + 1: step_size, :]
            feat_stacked.append(ef)
        feat_stacked = np.array(feat_stacked)

        self.feat_stacked = feat_stacked

    def extract_freq_band_envelope(self) -> np.ndarray:
        """
        Extracts frequency band envelopes for sEEG.

        Returns:
        -------
        np.ndarray
            Extracted frequency band envelopes.
        """
        self.logger.info(f'Extracting frequency band envlopes for sEEG')
        data = scipy.signal.detrend(self.eeg, axis=0)
        data = filter_data(data.T, self.sample_rate, 70, 170, method="iir").T
        data = filter_data(data.T, self.sample_rate, 102, 98, method="iir").T
        data = filter_data(data.T, self.sample_rate, 152, 148, method="iir").T

        data = np.abs(hilbert3(data))

        num_windows = int((data.shape[0] - self.window_size) / self.frame_shift) + 1

        feat = []

        for win in range(num_windows):
            start = win * self.frame_shift
            stop = start + self.window_size
            feat.append(data[start:stop, :])
        self.eeg_freq_band_features = np.array(feat)
        self.logger.info(f'Done Extracting frequency band envelopes for sEEG')

        pdb.set_trace()

        return self.eeg_freq_band_features

    def save_features(self, subject_id: str) -> None:
        """
        Saves the extracted features to a file.

        Parameters:
        ----------
        subject_id : str
            ID of the subject.
        """
        eeg_path = os.path.join(self.output_dir, f"P{subject_id}_eeg_features.npy")
        try:
            np.save(eeg_path, self.feat_stacked)
            self.logger.info(f"Saved feature: {eeg_path}")
        except Exception as e:
            self.logger.error(f"Failed to save feature to {eeg_path}: {e}")

        filename = f'P{subject_id}_eeg_features.npy'
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = Path(self.output_dir, filename)

        np.save(filepath, self.feat_stacked)

    def extract(self, subject: str) -> None:
        """
        Extracts features for a given subject.

        Parameters:
        ----------
        subject : str
            ID of the subject.
        """
        pdb.set_trace()
        self._load_eeg(subject_id=subject)

        self._clean_eeg()
        self._electrode_shaft_referencing()
        self.extract_freq_band_envelope()
        self.stack_features()


def extract_eeg_features() -> None:
    """
    Extracts EEG features for all subjects.
    """
    config_path = "configs/feature_extraction.yaml"
    extractor = EEGFeaturesExtractor(config_path)
    for subject_id in range(1, 31):
        subject_id = str(subject_id).zfill(2)
        extractor.extract(subject=subject_id)
        extractor.save_features(subject_id)
