import tensorflow as tf


def normalize_with_moments(x, axes):
    mean, variance = tf.nn.moments(x, axes=axes, name="normalization")
    return (x - mean) * 0.2 / tf.sqrt(variance)
