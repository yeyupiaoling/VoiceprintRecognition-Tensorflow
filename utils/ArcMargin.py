import tensorflow as tf


class ArcNet(tf.keras.Model):
    def __init__(self, feature_dim, n_classes, s=64.0, m=0.50, regularizer=tf.keras.regularizers.l2(5e-4), **kwargs):
        super(ArcNet, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.W = self.add_weight(shape=(feature_dim, self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 regularizer=self.regularizer)

    def call(self, x, y):
        x = tf.nn.l2_normalize(x, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)
        logits = x @ W

        # add margin
        theta = tf.acos(
            tf.keras.backend.clip(logits, -1.0 + tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()))
        target_logits = tf.cos(theta + self.m)

        y = tf.one_hot(tf.cast(y, tf.int32), depth=self.n_classes)
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)

        return out
