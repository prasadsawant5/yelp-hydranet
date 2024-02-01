import tensorflow as tf
from tensorflow.keras.layers import Dense


class Head:
    def __init__(self, scope_name: str, units: int = 8):
        self.scope_name = scope_name
        self.units = units

    def build_head(self, x):
        with tf.name_scope(self.scope_name):
            x = Dense(self.units, activation='relu', name=self.scope_name + '_dense')(x)
            x = Dense(2, activation='softmax', name=self.scope_name + '_output')(x)

        return x
