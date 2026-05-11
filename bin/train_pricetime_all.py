import traceback
import redis
from sqlalchemy import create_engine
from qf.dbsync.dbconfig import db_config
from qf.indicators import log_acceleration_direction
from qf.nn import set_seeds
from qf.nn.models import create_fractional_diff_model
from qf.nn.splitter import create_datasets
import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models
regularizers = tf.keras.regularizers
import os
import sys
import numpy as np
import pandas as pd

def read_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def shuffle(nd_array):
    return nd_array

def create_and_merge_datasets(quotes_file, engine, redis_connection, lookback_periods):
    quotes = read_lines(quotes_file)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = None, None, None, None, None, None
    with engine.connect() as connection:            
        for quote_name in quotes:
            try:
                if X_train is None:
                    X_train, X_val, X_test, Y_train, Y_val, Y_test, _ = create_datasets(log_acceleration_direction(connection, redis_connection, quote_name, lookback_periods))
                else:
                    X_train_extra, X_val_extra, X_test_extra, Y_train_extra, Y_val_extra, Y_test_extra, _ = create_datasets(log_acceleration_direction(connection, redis_connection, quote_name, lookback_periods))
                    X_train = np.concatenate((X_train, X_train_extra), axis=0)
                    X_val = np.concatenate((X_val, X_val_extra), axis=0)
                    X_test = np.concatenate((X_test, X_test_extra), axis=0)
                    Y_train = np.concatenate((Y_train, Y_train_extra), axis=0)
                    Y_val = np.concatenate((Y_val, Y_val_extra), axis=0)
                    Y_test = np.concatenate((Y_test, Y_test_extra), axis=0)
                    print(f"Loaded {quote_name}")
            except Exception as cause:
                print(f"Error backtesting {quote_name}: {cause}")
                traceback.print_exc()                    
        
        connection.close()
    return shuffle(X_train), shuffle(X_val), shuffle(X_test), shuffle(Y_train), shuffle(Y_val), shuffle(Y_test)

def remove_non_selected_files(selected_model_name):
    models_dir = os.path.join(os.getcwd(), 'models')
    for filename in os.listdir(models_dir):
        file_path = os.path.join(models_dir, filename)
        if os.path.isfile(file_path) and file_path != selected_model_name:
            os.remove(file_path)

if __name__ == "__main__":
    patience = 2
    epochs = 10
    batch_size = 100
    indicator_name = "price-time-wavelet-direction"
    quotes_file = sys.argv[1]
    min_lookback_periods = 14 if len(sys.argv) < 4 else int(sys.argv[2])
    max_lookback_periods = 30 if len(sys.argv) < 4 else int(sys.argv[3])
    input_dims = range(min_lookback_periods, max_lookback_periods + 1)
    set_seeds(42)
    _, sqlalchemy_url, redis_host, redis_port, redis_database = db_config()
    engine = create_engine(sqlalchemy_url)
    redis_connection = redis.Redis(host=redis_host, port=redis_port, db=redis_database, decode_responses=True)    
    best_mse = 0
    best_model_name = None

    for input_dim in input_dims:     
        X_train, X_val, X_test, Y_train, Y_val, Y_test = create_and_merge_datasets(quotes_file, engine, redis_connection, input_dim)
        for kernel_size in range(int(input_dim/4), int(input_dim/3) * 2 + 1):            
            checkpoint_filepath = os.path.join(os.getcwd(), 'models', f'{indicator_name}-{input_dim}-{kernel_size}.keras')
            model = create_fractional_diff_model(input_dim, kernel_size)
            model.summary()    
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
                
            best_model = tf.keras.models.load_model(checkpoint_filepath)
            mse, mae = best_model.evaluate(X_test, Y_test, verbose=0) 
            
            if best_model_name is None:
                best_model_name = checkpoint_filepath
                best_mse = mse
                print(f"First iteration, first model name = is {checkpoint_filepath}, mse = {mse}")
            elif mse < best_mse:
                best_model_name = checkpoint_filepath
                best_mse = mse
                print(f"Best model name = is {checkpoint_filepath}, mse = {mse}")
    
    # Remove all files in the models directory except the best model
    selected_model_name = best_model_name.replace(".keras", f"-{best_mse:.6f}.keras")
    os.rename(best_model_name, selected_model_name)
    engine.dispose()
    remove_non_selected_files(selected_model_name)
    
    