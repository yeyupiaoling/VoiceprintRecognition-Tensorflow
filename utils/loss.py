import math
from tensorflow.python.keras.utils import losses_utils
import tensorflow as tf


class ArcLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes, margin=0.5, scale=64, reduction=losses_utils.ReductionV2.AUTO):
        super().__init__(reduction=reduction)
        self.margin = margin
        self.scale = scale
        self.num_classes = num_classes
        self.threshold = tf.math.cos(math.pi - self.margin)
        self.cos_m = tf.math.cos(self.margin)
        self.sin_m = tf.math.sin(self.margin)
        self.mm = tf.multiply(self.sin_m, self.margin)

    def call(self, cos_theta, labels):
        sin_theta = tf.math.sqrt(1 - tf.math.square(cos_theta))
        cos_theta_m = tf.subtract(cos_theta * self.cos_m, sin_theta * self. sin_m)
        cos_theta_m = tf.where(cos_theta > self.threshold, cos_theta_m, cos_theta - self.mm)
        one_hot = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes)
        logits = tf.where(one_hot == 1., cos_theta_m, cos_theta)
        logits = tf.multiply(logits, self.scale)
        losses = tf.nn.softmax_cross_entropy_with_logits(one_hot, logits)
        return losses

    def get_config(self):
        config = super(ArcLoss, self).get_config()
        config.update({"margin": self.margin, "scale": self.scale})
        return config