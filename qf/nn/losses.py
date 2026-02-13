import numpy as np
import tensorflow as tf
register_keras_serializable = tf.keras.utils.register_keras_serializable

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

dynamic_penalty_weight = tf.Variable(2.0, trainable=False, dtype=tf.float32, name="dynamic_penalty_weight")

class PenaltyScheduler(tf.keras.callbacks.Callback):
    def __init__(self, start_val=2.0, end_val=20.0, total_epochs=75):
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