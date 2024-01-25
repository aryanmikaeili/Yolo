import torch
import numpy as np

import os

from PIL import Image
import pandas as pd

if __name__ == '__main__':
    annotations = pd.read_csv('./data/100examples.csv')
    image_dir = './data/images'
    label_dir = './data/labels'

    #get images and labels and save them in a folder

    new_data_dir = './small_data'
    os.makedirs(new_data_dir, exist_ok = True)

    new_image_dir = os.path.join(new_data_dir, 'images')
    new_label_dir = os.path.join(new_data_dir, 'labels')

    os.makedirs(new_image_dir, exist_ok = True)
    os.makedirs(new_label_dir, exist_ok = True)

    for idx in range(len(annotations)):
        image_path = os.path.join(image_dir, annotations.iloc[idx, 0])
        label_path = os.path.join(label_dir, annotations.iloc[idx, 1])



        #move label file which is a txt file to new folder
        os.system('cp {} {}'.format(label_path, new_label_dir))

        #move image file which is a png file to new folder
        os.system('cp {} {}'.format(image_path, new_image_dir))


