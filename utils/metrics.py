import tensorflow as tf
from tensorflow.keras.regularizers import l2


class ArcNet(tf.keras.layers.Layer):
    def __init__(self, num_classes, regularizer=l2(5e-4), **kwargs):
        super(ArcNet, self).__init__(**kwargs)
        self.regularizer = regularizer
        self.num_classes = num_classes

    def build(self, input_shape):
        self.weight = self.add_weight(shape=[input_shape[-1], self.num_classes],
                                      initializer=tf.keras.initializers.GlorotUniform(),
                                      trainable=True,
                                      regularizer=self.regularizer)

    def call(self, feature):
        normed_feature = tf.nn.l2_normalize(feature, axis=1)
        normed_weight = tf.nn.l2_normalize(self.weight, axis=0)
        cos_theta = tf.matmul(normed_feature, normed_weight)
        return cos_theta

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.num_classes,
                       "kernel_regularizer": self.regularizer})
        return config
