import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, Normalization, ReLU, ZeroPadding2D

class Stem:
    def __init__(self) -> None:
        self.k_size = 3
        self.scope_name = 'stem'
        self.filters = 32

    def build_stem(self, x: Input):
        with tf.name_scope(self.scope_name):
            x = Normalization(name='stem_normalization')(x)
            x = ZeroPadding2D(padding=(1, 1), name='stem_zero_padding')(x)
            x = Conv2D(self.filters, self.k_size, name='stem_conv')(x)
            x = BatchNormalization(name='stem_batch_norm')(x)
            x = ReLU(name='stem_act')(x)

        return x
