import os
import tensorflow as tf
import pandas as pd
from tensorflow.python.ops.gen_dataset_ops import MapDataset
from config import HEIGHT, WIDTH, LABELS

labels = pd.read_csv(LABELS)


def configure_for_performance(ds: MapDataset) -> MapDataset:
    ds = ds.cache()
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

# @tf.function
def process_input(img_path) -> tuple:
    # print()
    # print(type(img_path))
    # print(img_path)
    # print()
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, 3)

    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (HEIGHT, WIDTH))
    img = tf.clip_by_value(img, 0.0, 1.0)

    folder_name = tf.strings.split(img_path, os.sep)[-2]
    
    is_good_for_lunch = tf.strings.to_number(tf.strings.split(folder_name, sep='_')[1], tf.float32)
    is_good_for_dinner = tf.strings.to_number(tf.strings.split(folder_name, sep='_')[2], tf.float32)
    takes_reservations = tf.strings.to_number(tf.strings.split(folder_name, sep='_')[3], tf.float32)
    outdoor_seating = tf.strings.to_number(tf.strings.split(folder_name, sep='_')[4], tf.float32)
    is_expensive = tf.strings.to_number(tf.strings.split(folder_name, sep='_')[5], tf.float32)
    has_alcohol = tf.strings.to_number(tf.strings.split(folder_name, sep='_')[6], tf.float32)
    has_table_service = tf.strings.to_number(tf.strings.split(folder_name, sep='_')[7], tf.float32)
    ambience_is_classy = tf.strings.to_number(tf.strings.split(folder_name, sep='_')[8], tf.float32)
    good_for_kids = tf.strings.to_number(tf.strings.split(folder_name, sep='_')[9], tf.float32)

    # is_good_for_lunch = tf.strings.split(folder_name, sep='_')[1]
    # is_good_for_dinner = tf.strings.split(folder_name, sep='_')[2]
    # takes_reservations = tf.strings.split(folder_name, sep='_')[3]
    # outdoor_seating = tf.strings.split(folder_name, sep='_')[4]
    # is_expensive = tf.strings.split(folder_name, sep='_')[5]
    # has_alcohol = tf.strings.split(folder_name, sep='_')[6]
    # has_table_service = tf.strings.split(folder_name, sep='_')[7]
    # ambience_is_classy = tf.strings.split(folder_name, sep='_')[8]
    # good_for_kids = tf.strings.split(folder_name, sep='_')[9]

    return (img, is_good_for_lunch, is_good_for_dinner, takes_reservations, outdoor_seating, is_expensive, 
            has_alcohol, has_table_service, ambience_is_classy, good_for_kids)
