import os
import pdb
from pathlib import Path
import config as config
from src.logging.log import setup_logger, load_config

log_dir = Path(config.CUR_DIR, "logs")
log_path = Path(log_dir, "runner.log")
logger = setup_logger('runner', log_path)


if config.NPY_WAV_CONERSION:
    from src.audio.npy_to_audio import npy_to_wav_converter
    logger.info(f'Converting numpy files to wav files')
    npy_to_wav_converter()

if config.FEATURE_EXTRACTION:
    logger.info(f'Feature Extraction')
    from src.features.audio_to_mel_features import audio_to_mel_features
    from src.features.eeg_to_features import extract_eeg_features
    audio_to_mel_features()
    extract_eeg_features()








if config.TRAIN:
    from src.features.dataloader import EEGMelDataLoader
    from src.neural.trainner import ModelTrainer
    from src.utils.normalization import z_score_normalize

    trainer_path = Path(config.CUR_DIR, 'configs/trainer.yaml')
    train_config = load_config(trainer_path)
    
    subject = '01'

    dataloader = EEGMelDataLoader(config_path=Path(config.CUR_DIR, 'configs/feature_extraction.yaml'))
    eeg_feat, mel_feat = dataloader.load_subject(subject_id=subject)
    print(eeg_feat.shape, mel_feat.shape)

        

    input_shape = (eeg_feat.shape[1], eeg_feat.shape[2])
    model_dir = train_config.get('model_dir')

    if model_dir == 'NeuroIncept':
        from src.neural.models.neuroincept_decoder import NeuroInceptDecoder
        model = NeuroInceptDecoder(input_shape=input_shape, output_shape=128)
    
    if model == 'NeuralNetwork':
        from src.neural.models.neural_network import NeuralNetwork
        model = NeuralNetwork(input_shape=(16*127,), output_shape=128)
        
    X = z_score_normalize(eeg_feat)
    trainner = ModelTrainer(model_name=model_dir, subject_id=subject, model_dir=model_dir)
    trainner.train_model(model=model, X=X, y=mel_feat)


    