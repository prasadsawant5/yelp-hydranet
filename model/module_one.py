import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, ReLU, SeparableConv2D


class ModuleOne:
    def __init__(self, scope_name: str):
        self.scope_name = scope_name
        self.filters = 32
        self.k_size = 3
        self.padding = 'same'

    def build_module(self, x):
        with tf.name_scope(self.scope_name):
            x = SeparableConv2D(self.filters, self.k_size, padding=self.padding, name='block1a_dwconv')(x)
            x = BatchNormalization(name='block1a_bn')(x)
            x = ReLU(name='block1a_act')(x)

        return x

