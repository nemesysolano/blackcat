from qf.indicators import price_time_wavelet_force
import tensorflow as tf

from qf.nn.models.layers import FractionalDiffLayer
layers = tf.keras.layers
models = tf.keras.models
regularizers = tf.keras.regularizers
import numpy as np

def create_cnn_model(lookback_periods, indicator):
    # input_dim = 14 (your lags)
    inputs = layers.Input(shape=(lookback_periods, 1))
    # 1. Convolutional Layer: Scans for patterns using 32 different "filters"
    # kernel_size=3 means it looks at 3 consecutive lags at a time
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x) # Reduces noise
    
    # 2. Second Scan: Finds more complex combinations of the first patterns
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x) # Flattens the data for the final decision
    
    # 3. Final Decision Layers
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x) # Stabilizes learning
    x = layers.Dropout(0.1)(x)

    activation = 'linear' if indicator is price_time_wavelet_force else 'tanh'
  
    outputs = layers.Dense(1, activation = activation)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def create_fractional_diff_model(lookback_periods, indicator):
    # input_dim = 14 (your lags)
    inputs = layers.Input(shape=(lookback_periods, 1))
    # 1. Convolutional Layer: Scans for patterns using 32 different "filters"
    # kernel_size=3 means it looks at 3 consecutive lags at a time
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x) # Reduces noise
    
    # 2. Second Scan: Finds more complex combinations of the first patterns
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x) # Flattens the data for the final decision
    
    # 3. Final Decision Layers
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x) # Stabilizes learning
    x = layers.Dropout(0.1)(x)

    outputs = layers.Dense(1, activation = 'tanh')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model