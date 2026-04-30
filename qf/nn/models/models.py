import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models
regularizers = tf.keras.regularizers

def create_fractional_diff_model(input_dim):
    # input_dim = 14 (your lags)
    inputs = layers.Input(shape=(input_dim, 1))
    # 1. Convolutional Layer: Scans for patterns using 32 different "filters"
    # kernel_size=3 means it looks at 3 consecutive lags at a time
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x) # Reduces noise
    
    # 2. Second Scan: Finds more complex combinations of the first patterns
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x) # Flattens the data for the final decision
    
    # 3. Final Decision Layers
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
#   x = layers.BatchNormalization()(x) # Stabilizes learning
#   x = layers.Dropout(0.1)(x)

    outputs = layers.Dense(1, activation = 'tanh')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_normalized_ohlc_model(input_dim):
    """
    Creates an LSTM model to forecast the Normalized OHLC Bar Ƀ(t).
    
    Args:
        input_dim (int): The lookback window (k) for the sequence of past bars.
        
    Returns:
        tf.keras.Model: Compiled LSTM model expecting inputs of shape (batch, input_dim, 4)
                        and outputting predictions of shape (batch, 4).
    """
    # Each Ƀ(t-i) has 4 features: normalized đ1, đ2, đ3, and đ4
    inputs = layers.Input(shape=(input_dim, 4))
    
    # 1. Sequence Processing (LSTM)
    # First LSTM layer scans the temporal sequence to find structural patterns
    x = layers.LSTM(64, return_sequences=True, activation='tanh', kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x) # Stabilizes learning across the sequence
    x = layers.Dropout(0.2)(x)
    
    # Second LSTM layer condenses the sequence into a final state vector
    x = layers.LSTM(32, return_sequences=False, activation='tanh', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # 2. Dense Interpretation Layer
    # Interprets the LSTM state to map to the 4 distance geometries
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    
    # 3. Output Layer
    # 4 units corresponding to the predicted [đ1, đ2, đ3, đ4] vector at time t.
    # We use 'relu' because structural price distances * P(t) cannot be negative.
    outputs = layers.Dense(4, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    # MSE is standard for continuous distance regression
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model