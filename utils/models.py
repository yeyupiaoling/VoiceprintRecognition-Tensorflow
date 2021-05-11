import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten
from tensorflow.keras.applications import MobileNetV2, ResNet50
from .ArcMargin import ArcNet


class Model(tf.keras.Model):
    def __init__(self, input_shape, num_classes, feature_dim=512, backbone_type='ResNet50', margin=0.5, logist_scale=64,
                 embd_shape=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        regularizer1 = tf.keras.regularizers.l2(5e-4)
        regularizer2 = tf.keras.regularizers.l2(5e-4)
        self.net = None
        if backbone_type == 'ResNet50':
            self.net =  ResNet50(input_shape=input_shape, include_top=False, weights=None)
        elif backbone_type == 'MobileNetV2':
            self.net =  MobileNetV2(input_shape=input_shape, include_top=False, weights=None)
        else:
            raise TypeError('backbone_type error!')

        self.bn1 = BatchNormalization()
        self.dropout = Dropout(rate=0.5)
        self.flatten  = Flatten()
        self.fc = Dense(embd_shape, kernel_regularizer=regularizer1, name='embedding')
        self.bn2 = BatchNormalization()

        self.arc_net = ArcNet(feature_dim=feature_dim, n_classes=num_classes, m=margin, s=logist_scale, regularizer=regularizer2)

    def call(self, inputs, training=None, mask=None):
        if training:
            x, y = inputs
        else:
            x = inputs
        x = self.net(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        feature = self.bn2(x)

        if training:
            logits = self.arc_net(x=feature, y=y)
            return logits
        else:
            return feature