
from sqlalchemy import create_engine
from qf.indicators import price_time_wavelet_direction, price_time_wavelet_force, volume_time_wavelet_direction
from qf.dbsync import read_quote_names, db_config
from qf.nn import directional_mse
from qf.nn import PenaltyScheduler
from qf.nn.splitter import create_datasets
import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models
regularizers = tf.keras.regularizers
import os
import sys
import tensorflow as tf
import random
import numpy as np
import tensorflow as tf

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # Force TensorFlow to use deterministic ops (slower but reproducible)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def create_cnn_model(input_dim, indicator):
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
    x = layers.BatchNormalization()(x) # Stabilizes learning
    x = layers.Dropout(0.1)(x)

    activation = 'linear' if indicator is price_time_wavelet_force else 'tanh'
  
    outputs = layers.Dense(1, activation = activation)(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

indicator = {
    "price-time-wavelet-force": price_time_wavelet_force,
    "price-time-wavelet-direction": price_time_wavelet_direction,
    "volume-time-wavelet-direction": volume_time_wavelet_direction
}
if __name__ == "__main__":
    set_seeds(42)
    quote_name = sys.argv[1]
    indicator_name = sys.argv[2]
    scale_multiplier = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    indicator = indicator[indicator_name]
    lookback_periods = 14
    _, sqlalchemy_url = db_config()
    engine = create_engine(sqlalchemy_url)

    with engine.connect() as connection:
        print(f"Training {quote_name} with {indicator_name} indicator")
        X_train, X_val, X_test, Y_train, Y_val, Y_test, _ = create_datasets(indicator(connection, quote_name, lookback_periods))
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        model = create_cnn_model(X_train.shape[1], indicator)

        patience = 50
        epochs = 100
        batch_size = 50
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

        if indicator is price_time_wavelet_direction:
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
            callbacks = callbacks
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
                if indicator is price_time_wavelet_direction or indicator is volume_time_wavelet_direction:
                    print("Ticker,MSE,MAE,Match %,Different %,Pct Diff%,Edge,tradable", file=f)
                else:
                    print("Ticker,MSE,MAE", file=f)

            pct_diff = int(np.abs(matching_pct - different_pct) * 100)
            edge_diff = int((max(matching_pct, different_pct) * 100) - 50)
            tradable = edge_diff > 6
            if indicator is price_time_wavelet_direction or indicator is volume_time_wavelet_direction:
                print(f"{quote_name},{mse},{mae},{matching_pct * 100},{different_pct * 100},{pct_diff},{edge_diff},{tradable}", file=f)
            else:
                print(f"{quote_name},{mse},{mae}", file=f)
        print(f"Report saved to {output_file}")
        connection.close()
    engine.dispose()
    

