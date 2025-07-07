import os
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

if config.WAV_MEL_CONVERSION:
    logger.info(f'Converting wav files to mel spectrograms')
    from src.audio.audio_to_mel import AudioToMelConverter
    config_path = Path(os.getcwd(), "configs/wav_to_mel.yaml")
    converter = AudioToMelConverter(config_path)
    converter.convert()

if config.EEG_FEATURES_EXTRACTION:
    logger.info(f'Extracting EEG Features')
    from src.eeg.eeg_to_features import extract_eeg_features
    config_path = Path(config.CUR_DIR, 'configs/eeg_to_features.yaml')
    extract_eeg_features(
        config_path=config_path
    )








if config.TRAIN:
    from src.neural.models.neuroincept_decoder import NeuroInceptDecoder
    from src.neural.trainner import ModelTrainer
    from src.utils.normalization import z_score_normalize

    trainer_path = Path(config.CUR_DIR, 'configs/trainer.yaml')
    train_config = load_config(trainer_path)
    
    subject = '01'

    if train_config.get('load_eeg'):
        from src.neural.data.eeg_data_loader import EEGDataLoader
        
        logger.info(f'Loading EEG features')
        dataloader = EEGDataLoader(subject=subject)
        eeg_feat = dataloader.get_eeg_features()
    
    if train_config.get('load_mel'):
        from src.neural.data.audio_data_loader import AudioDataLoader
        
        logger.info(f'Loading Mel Spectropgrams')
        dataloader = AudioDataLoader(subject=subject)
        mel_feat = dataloader.get_mels().T

    if train_config.get('load_eeg') and  train_config.get('load_mel'):    
        print(eeg_feat.shape, mel_feat.shape)
        min_len = min(len(mel_feat), len(eeg_feat))
        mel_feat = mel_feat[:min_len]
        eeg_feat = eeg_feat[:min_len]
        print(eeg_feat.shape, mel_feat.shape)

        input_shape = (eeg_feat.shape[1], eeg_feat.shape[2])
        model_name =  train_config.get('model')
        model_dir = train_config.get('model_dir')
        
        if model_name == 'NeuroIncept':
            model = NeuroInceptDecoder(input_shape=input_shape, output_shape=128)
        if model_name == 'NeuralNetwork':
            from src.neural.models.neural_network import NeuralNetwork
            model = NeuralNetwork(input_shape=(16*127,), output_shape=128)
        
        X = z_score_normalize(eeg_feat)
        trainner = ModelTrainer(model_name=model_name, subject_id=subject, model_dir=model_dir)
        trainner.train_model(model=model, X=X, y=mel_feat)



    else:
        print('Set eeg and mel loding to True')


    