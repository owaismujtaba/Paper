import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

class NeuralNetwork:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def _create_model(self):
        self.model = keras.Sequential([
            layers.Input(shape=self.input_shape),

            # Reduce number of layers and add batch normalization
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            #layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            #layers.BatchNormalization(),
            #layers.Dropout(0.1),

            layers.Dense(self.output_shape, activation=None)  # Linear activation for regression
        ])
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='mse')

    def train(self, X_train, y_train):
        self._create_model()
        X_train = X_train.reshape(X_train.shape[0], -1)
        y_train = y_train.reshape(y_train.shape[0], -1)

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        self.model.fit(X_train, y_train,
                       batch_size=32, 
                       epochs=100, 
                       validation_split=0.10,
                       callbacks=[early_stopping])