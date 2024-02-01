import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from weighted_cross_entropy import WeightedCrossEntropy


class Trainer:
    def __init__(self):
        self.good_for_lunch_loss = WeightedCrossEntropy('good_for_lunch_loss')
        self.good_for_dinner_loss = WeightedCrossEntropy('good_for_dinner_loss')
        self.takes_reservation_loss = WeightedCrossEntropy('takes_reservation_loss')
        self.outdoor_seating_loss = WeightedCrossEntropy('outdoor_seating_loss')
        self.expensive_loss = WeightedCrossEntropy('expensive_loss')
        self.alcohol_loss = WeightedCrossEntropy('alcohol_loss')
        self.table_service_loss = WeightedCrossEntropy('table_service_loss')
        self.classy_ambience_loss = WeightedCrossEntropy('classy_ambience_loss')
        self.good_for_kids_loss = WeightedCrossEntropy('good_for_kids_loss')

    @tf.function
    def train_step(self, model: Model, opt: Adam, img, is_good_for_lunch, is_good_for_dinner, takes_reservations,
                   outdoor_seating, is_expensive, has_alcohol, has_table_service, ambience_is_classy, good_for_kids) -> tuple:
        with tf.GradientTape() as tape:
            (lunch_logits, dinner_logits, reservation_logits, outdoor_seating_logits, expensive_logits,
             alcohol_logits, table_service_logits, ambience_logits, kids_logits) = model(img, training=True)

            lunch_loss_val = self.good_for_lunch_loss(is_good_for_lunch, lunch_logits)
            dinner_loss_val = self.good_for_dinner_loss(is_good_for_dinner, dinner_logits)
            reservation_loss_val = self.takes_reservation_loss(takes_reservations, reservation_logits)
            outdoor_seating_loss_val = self.outdoor_seating_loss(outdoor_seating, outdoor_seating_logits)
            expensive_loss_val = self.expensive_loss(is_expensive, expensive_logits)
            alcohol_loss_val = self.alcohol_loss(has_alcohol, alcohol_logits)
            table_service_loss_val = self.table_service_loss(has_table_service, table_service_logits)
            ambience_loss_val = self.classy_ambience_loss(ambience_is_classy, ambience_logits)
            kids_loss_val = self.good_for_kids_loss(good_for_kids, kids_logits)

            total_loss = tf.math.reduce_mean([
                lunch_loss_val, dinner_loss_val, reservation_loss_val, outdoor_seating_loss_val, expensive_loss_val,
                alcohol_loss_val, table_service_loss_val, ambience_loss_val, kids_loss_val
            ])

        total_grads = tape.gradient(total_loss, model.trainable_weights)
        # dinner_grads = tape.gradient(dinner_loss_val, model.trainable_weights)
        # reservation_grads = tape.gradient(reservation_loss_val, model.trainable_weights)
        # outdoor_grads = tape.gradient(outdoor_seating_loss_val, model.trainable_weights)
        # expensive_grads = tape.gradient(expensive_loss_val, model.trainable_weights)
        # alcohol_grads = tape.gradient(alcohol_loss_val, model.trainable_weights)
        # table_grads = tape.gradient(table_service_loss_val, model.trainable_weights)
        # ambience_grads = tape.gradient(ambience_loss_val, model.trainable_weights)
        # kids_grads = tape.gradient(kids_loss_val, model.trainable_weights)

        opt.apply_gradients(zip(total_grads, model.trainable_weights))
        # dinner_opt.apply_gradients(zip(dinner_grads, model.trainable_weights))
        # reservation_opt.apply_gradients(zip(reservation_grads, model.trainable_weights))
        # seating_opt.apply_gradients(zip(outdoor_grads, model.trainable_weights))
        # expensive_opt.apply_gradients(zip(expensive_grads, model.trainable_weights))
        # alcohol_opt.apply_gradients(zip(alcohol_grads, model.trainable_weights))
        # table_opt.apply_gradients(zip(table_grads, model.trainable_weights))
        # ambience_opt.apply_gradients(zip(ambience_grads, model.trainable_weights))
        # kids_opt.apply_gradients(zip(kids_grads, model.trainable_weights))

        return (lunch_logits, dinner_logits, reservation_logits, outdoor_seating_logits, expensive_logits,
                alcohol_logits, table_service_logits, ambience_logits, kids_logits, total_loss)
