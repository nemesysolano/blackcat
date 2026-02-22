
import os
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