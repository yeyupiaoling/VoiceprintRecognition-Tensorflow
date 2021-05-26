import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2, ResNet50


class Model(tf.keras.Model):
    def __init__(self, input_shape, backbone_type='ResNet50', feature_dim=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        regularizer = tf.keras.regularizers.l2(5e-4)
        self.net = None
        if backbone_type == 'ResNet50':
            self.net = ResNet50(input_shape=input_shape, include_top=False, weights=None, pooling='max')
        elif backbone_type == 'MobileNetV2':
            self.net = MobileNetV2(input_shape=input_shape, include_top=False, weights=None, pooling='max')
        else:
            raise TypeError('backbone_type error!')
        self.bn1 = BatchNormalization()
        self.dropout = Dropout(rate=0.5)
        self.fc = Dense(feature_dim, kernel_regularizer=regularizer, bias_initializer='glorot_uniform')
        self.bn2 = BatchNormalization()

    def call(self, x):
        x = self.net(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc(x)
        feature = self.bn2(x)
        return feature
