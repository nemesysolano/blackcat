
from qf.indicators import scale_with_multiplier
from qf.indicators import volumeprice
from qf.indicators import wavelets
from qf.dbsync import read_quote_names, db_config
from qf.nn.splitter import create_datasets
import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models
regularizers = tf.keras.regularizers
import os
import sys
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import register_keras_serializable

# Global variable to track the penalty weight (starts at 2.0)
dynamic_penalty_weight = tf.Variable(2.0, trainable=False, dtype=tf.float32, name="dynamic_penalty_weight")

@register_keras_serializable(package='Custom', name='directional_mse')
def directional_mse(y_true, y_pred):
    mse = tf.square(y_true - y_pred)
    
    # 1. Sign Penalty (The existing logic)
    sign_mismatch = tf.cast(tf.sign(y_true) != tf.sign(y_pred), tf.float32)
    
    # 2. Conviction Penalty (The MAGIC)
    # If |pred| < |true| * 0.5, the model is being "cowardly"
    too_weak = tf.cast(tf.abs(y_pred) < (tf.abs(y_true) * 0.5), tf.float32)
    
    # 3. Combine penalties
    # We use a very high weight (e.g., 15.0) for wrong direction
    # and a medium weight (e.g., 3.0) for being too weak
    total_penalty = (sign_mismatch * dynamic_penalty_weight) + (too_weak * 3.0)
    
    return tf.reduce_mean(mse * (1.0 + total_penalty))

class PenaltyScheduler(tf.keras.callbacks.Callback):
    def __init__(self, start_val=2.0, end_val=10.0, total_epochs=50):
        super().__init__()
        self.start_val = start_val
        self.end_val = end_val
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        # Linear increase from start_val to end_val over the course of training
        new_val = self.start_val + (self.end_val - self.start_val) * (epoch / self.total_epochs)
        new_val = min(new_val, self.end_val) # Cap at the end value
        
        # Set the value in the backend so the loss function sees it
        tf.keras.backend.set_value(dynamic_penalty_weight, new_val)
        print(f" - [Callback] Penalty weight for epoch {epoch+1} is {new_val:.2f}")

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Force TensorFlow to use deterministic ops (slower but reproducible)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def create_cnn_model(input_dim):
    # input_dim = 14 (your lags)
    inputs = layers.Input(shape=(input_dim, 1))

    # 1. Convolutional Layer: Scans for patterns using 32 different "filters"
    # kernel_size=3 means it looks at 3 consecutive lags at a time
    x = layers.Conv1D(filters=32, kernel_size=3, activation='leaky_relu', padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x) # Reduces noise
    
    # 2. Second Scan: Finds more complex combinations of the first patterns
    x = layers.Conv1D(filters=64, kernel_size=3, activation='leaky_relu', padding='same')(x)
    x = layers.GlobalAveragePooling1D()(x) # Flattens the data for the final decision
    
    # 3. Final Decision Layers
    x = layers.Dense(32, activation='leaky_relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='tanh')(x) 

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=directional_mse, metrics=['mae'])
    return model

indicator = {
    "pricetime": wavelets.pricetime,
    "volumetime": wavelets.volumetime,
    "volumeprice": volumeprice
}
if __name__ == "__main__":
    set_seeds(42)
    quote_name = sys.argv[1]
    indicator_name = sys.argv[2]
    scale_multiplier = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    indicator = indicator[indicator_name]
    lookback_periods = 14
    _, sqlalchemy_url = db_config()
    
    X_train, X_val, X_test, Y_train, Y_val, Y_test = create_datasets(scale_with_multiplier(indicator(sqlalchemy_url, quote_name, lookback_periods), scale_multiplier))

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    model = create_cnn_model(X_train.shape[1])
    patience = 10
    epochs = 50
    batch_size = 32
    model.summary()
    checkpoint_filepath = os.path.join(os.getcwd(), 'models', f'{quote_name}-{indicator_name}.keras')

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        monitor='val_mae',
        mode='min'
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_mae',
        mode='min',
        restore_best_weights=True,
        patience = patience
    )

    # Instantiate the new scheduler
    penalty_callback = PenaltyScheduler(start_val=2.0, end_val=12.0, total_epochs=50)

    if indicator is volumeprice:
        callbacks=(
            model_checkpoint_callback, 
            early_stopping_callback,
            penalty_callback
        )        
    else:
        callbacks=(
            model_checkpoint_callback, 
            early_stopping_callback
        )                
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=(
            model_checkpoint_callback, 
            early_stopping_callback,
            # penalty_callback
        )
    )    

  
    best_model = tf.keras.models.load_model(checkpoint_filepath, custom_objects={'directional_mse': directional_mse})
    mse, mae = best_model.evaluate(X_test, Y_test, verbose=0) 
    
    # 3. Apply Polarity to Test Set
    Y_pred_raw = best_model.predict(X_test).flatten() 
    Y_pred = np.int32(np.sign(Y_pred_raw)).flatten()
    Y_expected = np.int32(np.sign(Y_test)).flatten()

    matching  = Y_pred == Y_expected
    different = Y_pred != Y_expected
    matching_pct = np.count_nonzero(matching) / len(Y_pred)
    different_pct = np.count_nonzero(different) / len(Y_pred)

    print(f"Finished with {quote_name} Model Training")
    output_file = os.path.join(os.getcwd(), "test-results", f"report-{indicator_name}.csv")
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode) as f:
        if mode == 'w':
            print("Ticker,MSE,MAE,Match %,Diff %,Pct Diff%,Edge,tradable", file=f)
        pct_diff = int(np.abs(matching_pct - different_pct) * 100)
        edge_diff = int((max(matching_pct, different_pct) * 100) - 50)
        tradable = edge_diff > 6
        print(f"{quote_name},{mse:.4f},{mae:.4f},{matching_pct:.4f},{different_pct:.4f},{pct_diff},{edge_diff},{tradable}", file=f)
