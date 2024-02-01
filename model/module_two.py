import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, ReLU, SeparableConv2D, ZeroPadding2D


class ModuleTwo:
    def __init__(self, scope_name: str):
        self.scope_name = scope_name
        self.filters = 96
        self.k_size = 3

    def build_model(self, x):
        with tf.name_scope(self.scope_name):
            x = SeparableConv2D(self.filters, kernel_size=self.k_size, name='dw_conv0')(x)
            x = BatchNormalization(name='bn0')(x)
            x = ReLU(name='act0')(x)
            x = ZeroPadding2D(padding=1, name='zero_padding')(x)
            x = SeparableConv2D(self.filters, kernel_size=self.k_size, name='dw_conv1')(x)
            x = BatchNormalization(name='bn1')(x)
            x = ReLU(name='act1')(x)

        return x
    