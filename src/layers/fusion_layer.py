import tensorflow as tf


class FusionLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs) -> None:
        super(FusionLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.free_embeddings = tf.Variable(
            tf.random_normal_initializer(mean=0.0, stddev=0.01)(shape=input_shape, dtype=tf.float32),
            trainable=True,
            name="free_embeddings",
        )

    def call(self, inputs):
        return inputs + self.free_embeddings
