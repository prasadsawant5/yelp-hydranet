import tensorflow as tf
from tensorflow.keras.losses import Loss, SparseCategoricalCrossentropy
from config import BATCH_SIZE


class WeightedCrossEntropy(Loss):
    def __init__(self, name: str = 'loss'):
        super().__init__(name=name)
        self.scce = SparseCategoricalCrossentropy()

    def call(self, y_true, y_pred):
        weight_pos = (BATCH_SIZE - tf.math.reduce_sum(y_true)) / BATCH_SIZE
        loss = self.scce(y_true, y_pred)
        return tf.math.multiply(weight_pos, loss)
