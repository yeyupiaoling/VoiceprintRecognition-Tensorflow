import math

import tensorflow as tf


class ArcLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, margin=0.5, scale=64):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.num_classes = num_classes
        self.threshold = tf.math.cos(math.pi - self.margin)
        self.cos_m = tf.math.cos(self.margin)
        self.sin_m = tf.math.sin(self.margin)
        self.mm = tf.multiply(self.sin_m, self.margin)

    # @tf.function
    def call(self, cos_theta, labels):
        sin_theta = tf.math.sqrt(1 - tf.math.square(cos_theta))
        cos_theta_m = tf.subtract(cos_theta * self.cos_m, sin_theta * self. sin_m)
        cos_theta_m = tf.where(cos_theta > self.threshold, cos_theta_m, cos_theta - self.mm)
        one_hot = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes)
        logits = tf.where(one_hot == 1., cos_theta_m, cos_theta)
        logits = tf.multiply(logits, self.scale)
        logits = tf.nn.softmax(logits)
        losses = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
        return losses

    def get_config(self):
        config = super(ArcLoss, self).get_config()
        config.update({"margin": self.margin, "scale": self.scale})
        return config