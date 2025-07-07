import os
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


import pdb
import config as config
from src.logging.log import setup_logger
from src.utils.metrics import calculate_pcc_spectrgorams

class ModelTrainer:
    def __init__(self, model_name, subject_id, model_dir, val_size=0.15):
        print("📊 Initializing ModelTrainer Class")
        self.name = model_name
        self.subjet_id = subject_id
        self.val_size = val_size
        self.dir = Path(config.CUR_DIR, model_dir)
        self.model_dir = Path(self.dir, self.subjet_id , model_name)
        self.model_path = Path(self.model_dir, f'{model_name}.h5')
        os.makedirs(self.model_dir, exist_ok=True)

        
        print("✅ ModelTrainer Initialization Complete ✅")

    def train_model(self, model, X, y, k=5):
        self.model = model
        print(f"🔧 Starting Model Training with {k}-Fold Cross Validation 🔧")
        print(f"🟢 Initial Data Shapes: X={X.shape}, y={y.shape}")

        kf = KFold(n_splits=k, shuffle=False)  # K-Fold CV
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\n🔄 Fold {fold + 1}/{k} in K-Fold CV")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            print(train_idx)
            print(val_idx)

            history = self.model.train(X_train, y_train)

            try:
                history_path = Path(self.model_dir, 'history.csv')
                history.to_csv(history_path)
                print(f"💾 Training history saved at: {history_path}")

                model.save(self.model_path)
                print(f"💾 Model saved at: {self.model_path}")
            except:
                print('⚠️ History saving not allowed')
                self.model_type = 'Reg'

            score = self.evaluate_model(X_val, y_val, fold)

            print(f"✅ Fold {fold + 1} Score: {score}")
            fold_scores.append(score)
            

        fold_scores = np.array(fold_scores)
        avg_mse, avg_rmse, avg_r2, avg_pcc = np.mean(fold_scores, axis=0)

        print('-'*60)
        print( f'Subject ID:  {self.subjet_id}')
        print( '-'*60)


        print("\n📊 Final Cross-Validation Results:")
        print(f"🔹 Average RMSE: {avg_rmse:.4f}")
        print(f"🔹 Average MSE: {avg_mse:.4f}")
        print(f"🔹 Average R² Score: {avg_r2:.4f}")
        print(f"🔹 Average PCC: {avg_pcc:.4f}")
        print('-'*60)  
        print('-'*60)

   
    def evaluate_model(self, X, y, fold):
        print("🔍 Evaluating Model 🔍")
        print(f"🟢 Input Data Shapes: X={X.shape}, y={y.shape}")
        if self.model_type =='Reg':
            predictions = self.model.model.predict(X)
        else:
            predictions = self.model.predict(X)
        print(f"📊 Predictions Shape: {predictions.shape}")
        predicted_flat = predictions.flatten()
        y_flatten = y.flatten()
        mse = mean_squared_error(y_flatten, predicted_flat)
        '''avergate the correlations across time and then avergate across samples'''
        rmse = np.sqrt(mse)
        r2 = r2_score(y_flatten, predicted_flat)
        pcc = calculate_pcc_spectrgorams(predictions, y)

        print(f"📊 RMSE: {rmse}, MSE: {mse}, 'R2: {r2}, PCC: {pcc}")

        np.save(str(Path(self.model_dir, f'Fold_{fold}_metrics.npy')), np.array([mse, rmse, r2, pcc]))
        self.metrices = [mse, rmse, r2, pcc]
        print(f"💾 Metrics values saved at: {str(Path(self.model_dir, f'Fold_{fold}_metrics.npy'))}")
        
        #plot_spectrograms(y, predictions)

        output_dir = Path(config.CUR_DIR, 'Predictions', str(self.subjet_id), self.name)
        pred_filepath = Path(output_dir, f'Predictions_Fold_{fold}.npy')
        actual_filepath = Path(output_dir, f'Actual_Fold_{fold}.npy')

        os.makedirs(output_dir, exist_ok=True)
        np.save(pred_filepath, predictions)
        np.save(actual_filepath, y)

        return self.metrices