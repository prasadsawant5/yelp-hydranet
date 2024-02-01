import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras import Model
from preprocess import process_input
from config import BATCH_SIZE, DATA, EPOCHS, HEIGHT, LOGS, LR, WIDTH
from model.heads.head import Head
from trainer import Trainer

if __name__ == '__main__':
    if not os.path.exists(LOGS):
        os.mkdir(LOGS)

    AUTO = tf.data.AUTOTUNE

    trainer = Trainer()

    train_ds = tf.data.Dataset.list_files(DATA + '/*/*', shuffle=True)
    train_ds = train_ds.map(process_input, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO)

    train_summary_writer = tf.summary.create_file_writer(LOGS)

    IMG_SHAPE = (HEIGHT, WIDTH, 3)
    backbone = tf.keras.applications.EfficientNetB0(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
    backbone.trainable = False

    inputs = Input(IMG_SHAPE, name='inputs')
    x = backbone(inputs, training=False)
    x = Conv2D(256, kernel_size=1, activation='relu', kernel_initializer='he_uniform', name='1x1_conv')(x)
    x = Flatten(name='flatten')(x)

    good_for_lunch_pred = Head('good_for_lunch_head').build_head(x)
    good_for_dinner_pred = Head('good_for_dinner_head').build_head(x)
    takes_reservation_pred = Head('takes_reservation_head').build_head(x)
    outdoor_seating_pred = Head('outdoor_seating_head').build_head(x)
    expensive_pred = Head('expensive_head').build_head(x)
    alcohol_pred = Head('alcohol_head').build_head(x)
    table_service_pred = Head('table_service_head').build_head(x)
    classy_ambience_pred = Head('classy_ambience_head').build_head(x)
    good_for_kids_pred = Head('good_for_kids_head').build_head(x)

    opt = Adam(LR, name='opt')
    # dinner_opt = Adam(LR, name='dinner_opt')
    # reservation_opt = Adam(LR, name='reservation_opt')
    # seating_opt = Adam(LR, name='seating_opt')
    # expensive_opt = Adam(LR, name='expensive_opt')
    # alcohol_opt = Adam(LR, name='alcohol_opt')
    # table_opt = Adam(LR, name='table_opt')
    # ambience_opt = Adam(LR, name='ambience_opt')
    # kids_opt = Adam(LR, name='kids_opt')

    lunch_metric = SparseCategoricalAccuracy(name='lunch_acc')
    dinner_metric = SparseCategoricalAccuracy(name='dinner_acc')
    reservation_metric = SparseCategoricalAccuracy(name='reservation_acc')
    seating_metric = SparseCategoricalAccuracy(name='seating_acc')
    expensive_metric = SparseCategoricalAccuracy(name='expensive_acc')
    alcohol_metric = SparseCategoricalAccuracy(name='alcohol_acc')
    table_metric = SparseCategoricalAccuracy(name='table_acc')
    ambience_metric = SparseCategoricalAccuracy(name='ambience_acc')
    kids_metric = SparseCategoricalAccuracy(name='kids_acc')

    model = Model(inputs=[inputs], outputs=[good_for_lunch_pred, good_for_dinner_pred, takes_reservation_pred,
                                   outdoor_seating_pred, expensive_pred, alcohol_pred, table_service_pred,
                                   classy_ambience_pred, good_for_kids_pred]
                  )

    model.summary()

    for epoch in range(EPOCHS):
        total_loss = None
        for (i, batch) in train_ds.enumerate():
            img = batch[0]
            is_good_for_lunch = batch[1]
            is_good_for_dinner = batch[2]
            takes_reservations = batch[3]
            outdoor_seating = batch[4]
            is_expensive = batch[5]
            has_alcohol = batch[6]
            has_table_service = batch[7]
            ambience_is_classy = batch[8]
            good_for_kids = batch[9]

            (lunch_logits, dinner_logits, reservation_logits, outdoor_seating_logits, expensive_logits,
             alcohol_logits, table_service_logits, ambience_logits, kids_logits, total_loss) = trainer.train_step(
                model, opt, img, is_good_for_lunch, is_good_for_dinner, takes_reservations, outdoor_seating,
                is_expensive, has_alcohol, has_table_service, ambience_is_classy, good_for_kids
            )

            lunch_metric.update_state(is_good_for_lunch, lunch_logits)
            dinner_metric.update_state(is_good_for_dinner, dinner_logits)
            reservation_metric.update_state(takes_reservations, reservation_logits)
            seating_metric.update_state(outdoor_seating, outdoor_seating_logits)
            expensive_metric.update_state(is_expensive, expensive_logits)
            alcohol_metric.update_state(has_alcohol, alcohol_logits)
            table_metric.update_state(has_table_service, table_service_logits)
            ambience_metric.update_state(ambience_is_classy, ambience_logits)
            kids_metric.update_state(good_for_kids, kids_logits)

        lunch_acc = lunch_metric.result()
        dinner_acc = dinner_metric.result()
        reservation_acc = reservation_metric.result()
        outdoor_acc = seating_metric.result()
        expensive_acc = expensive_metric.result()
        alcohol_acc = alcohol_metric.result()
        table_acc = table_metric.result()
        ambience_acc = ambience_metric.result()
        kids_acc = kids_metric.result()

        with train_summary_writer.as_default():
            tf.summary.scalar('lunch_acc', lunch_acc, step=epoch)
            tf.summary.scalar('dinner_acc', dinner_acc, step=epoch)
            tf.summary.scalar('reservation_acc', reservation_acc, step=epoch)
            tf.summary.scalar('outdoor_acc', outdoor_acc, step=epoch)
            tf.summary.scalar('expensive_acc', expensive_acc, step=epoch)
            tf.summary.scalar('alcohol_acc', alcohol_acc, step=epoch)
            tf.summary.scalar('table_acc', table_acc, step=epoch)
            tf.summary.scalar('ambience_acc', ambience_acc, step=epoch)
            tf.summary.scalar('kids_acc', kids_acc, step=epoch)

        lunch_metric.reset_states()
        dinner_metric.reset_states()
        reservation_metric.reset_states()
        seating_metric.reset_states()
        expensive_metric.reset_states()
        alcohol_metric.reset_states()
        table_metric.reset_states()
        ambience_metric.reset_states()
        kids_metric.reset_states()

        print('Epoch: {:03d}, Total Loss: {:.3f} Lunch Acc: {:.3f}, Dinner Acc: {:.3f}, Res Acc: {:.3f}, '.format(epoch + 1, total_loss, lunch_acc, dinner_acc, reservation_acc), end='')
        print('Outdoor Acc: {:.3f}, Expensive Acc: {:.3f}, Alcohol Acc: {:.3f}, Table Acc: {:.3f}, Ambience Acc: {:.3f}, Kids Acc: {:.3f}'.format(outdoor_acc, expensive_acc, alcohol_acc, table_acc, ambience_acc, kids_acc))

    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
