import tensorflow as tf
import numpy as np
layers = tf.keras.layers
models = tf.keras.models
regularizers = tf.keras.regularizers
constraints = tf.keras.constraints

register_keras_serializable = tf.keras.utils.register_keras_serializable

@register_keras_serializable(package='Custom', name='fractional_diff_layer')
class FractionalDiffLayer(layers.Layer):
    def __init__(self, window_size, **kwargs):
        super(FractionalDiffLayer, self).__init__(**kwargs)
        self.N = window_size
        # Constraint ensures 0 < s < 1 as per README
        self.s = self.add_weight(
            name="s_order",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(0.5),
            constraint=constraints.MinMaxNorm(min_value=0.01, max_value=0.99),
            trainable=True
        )
