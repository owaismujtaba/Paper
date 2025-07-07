import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import GRU, Input, Conv1D, MaxPooling1D, concatenate, Dense, Flatten, Reshape

import pdb

import config as config

early_stopping = EarlyStopping(
    monitor='val_loss',   
    patience=5,    
    restore_best_weights=True, 
    verbose=1
)


class NeuroInceptDecoder:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        

    def inception_module(self, input_tensor, filters):
        conv_1x1 = Conv1D(filters, 1, padding='same', activation='relu')(input_tensor)
        conv_3x3 = Conv1D(filters, 3, padding='same', activation='relu')(input_tensor)
        conv_5x5 = Conv1D(filters, 5, padding='same', activation='relu')(input_tensor)

        max_pool = MaxPooling1D(3, strides=1, padding='same')(input_tensor)
        max_pool = Conv1D(filters, 1, padding='same', activation='relu')(max_pool)

        output = concatenate([conv_1x1, conv_3x3, conv_5x5, max_pool], axis=-1)
        
        return output

    def create_model(self):
        input_layer = Input(shape=self.input_shape)

        # Inception Module 1
        x = self.inception_module(input_layer, 64)

        # GRU Module
        x = GRU(64, return_sequences=True)(x)
        x = GRU(128, return_sequences=True)(x)
        x = GRU(128, return_sequences=False)(x)

        x = Reshape((1, 128))(x)

        # Inception Module 2
        x = self.inception_module(x, 128)

        x = Flatten()(x)

        # Fully Connected Layers
        x = Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.BatchNormalization()(x)
        x = Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.BatchNormalization()(x)
        x = Dense(128, activation='relu')(x)
        
        

        output_layer = Dense(self.output_shape, activation='linear')(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, X_train, y_train):
        self.model = self.create_model()
        
        #X_train = X_train.reshape(X_train.shape[0], self.input_shape[0], 1)
        #y_train = y_train.reshape(y_train.shape[0], -1)
        self.model.fit(X_train, y_train,
            batch_size=32, 
            epochs=10, 
            validation_split=0.10,
            callbacks=[early_stopping]
        )
