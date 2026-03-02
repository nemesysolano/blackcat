import tensorflow as tf
import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models
regularizers = tf.keras.regularizers
constraints = tf.keras.constraints

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

    def get_fractional_weights(self):
        # Recursive formula: w_k = w_{k-1} * (k - 1 - s) / k
        indices = tf.range(1, self.N + 1, dtype=tf.float32)
        
        # We start weights with w_0 = 1 (not used in forecast model)
        # But for the forecast sum from k=1 to N:
        weights = [tf.constant([-1.0]) * self.s] # w_1 = -s
        for k in range(2, self.N + 1):
            prev_w = weights[-1]
            current_w = prev_w * (float(k - 1) - self.s) / float(k)
            weights.append(current_w)
        
        return tf.concat(weights, axis=0)

    def call(self, inputs):
        # inputs shape: (batch, window_size) -> {Z(t-N), ..., Z(t-1)}
        w = self.get_fractional_weights()
        # Flip weights to align with {Z(t-1), ..., Z(t-N)} order
        w_reversed = tf.reverse(w, axis=[0])
        
        # Matrix multiplication to get the sum from k=1 to N
        return tf.reduce_sum(inputs * w_reversed, axis=1, keepdims=True)