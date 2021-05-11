import tensorflow as tf


class FocalLoss(tf.keras.Model):

    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = tf.keras.losses.CategoricalCrossentropy()

    def call(self, input, target):
        logp = self.ce(input, target)
        p = tf.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
