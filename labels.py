import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import DATA, LABELS, TRAIN_PHOTOS_TO_BIZ_IDS, TRAIN, TRAIN_PHOTOS

if __name__ == '__main__':
    if not os.path.exists(DATA):
        os.mkdir(DATA)
    photos_to_biz = pd.read_csv(TRAIN_PHOTOS_TO_BIZ_IDS)
    biz_df = pd.read_csv(TRAIN)
    biz_df.set_index('business_id', inplace=True)

    columns = ['image_id', 'is_good_for_lunch', 'is_good_for_dinner', 'takes_reservations', 'outdoor_seating', 'is_expensive', 'has_alcohol', 
               'has_table_service', 'ambience_is_classy', 'good_for_kids']

    labels = []

    for i in tqdm(range(1, photos_to_biz.shape[0])):
        img_id = photos_to_biz.loc[i]['photo_id']
        biz_id = photos_to_biz.loc[i]['business_id']

        label = str((biz_df.loc[biz_id][0])).split(' ')
        row = [str(img_id) + '.jpg',] + [0 if str(i) not in label else 1 for i in range(0, len(columns[1:]))]

        folder_name = str(img_id)
        for i in range(0, len(columns[1:])):
            if str(i) not in label:
                folder_name += '_0'
            else:
                folder_name += '_1'

        img_path = os.path.join(DATA, folder_name)

        if not os.path.exists(img_path):
            os.mkdir(img_path)

        src = os.path.join(TRAIN_PHOTOS, str(img_id) + '.jpg')
        dest = os.path.join(img_path, str(img_id) + '.jpg')
        
        shutil.copyfile(src, dest)

        labels.append(row)

    labels_df = pd.DataFrame(labels, columns=columns)
    labels_df.set_index('image_id', inplace=True)
    labels_df.to_csv(LABELS)
    