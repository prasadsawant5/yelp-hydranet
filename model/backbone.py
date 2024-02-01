import tensorflow as tf
from tensorflow.keras.layers import Input
from stem import Stem


class Backbone:
    def __init__(self):
        self.stem = Stem()

    def build_backbone(self, x: Input):
        x = self.stem.build_stem(x)


