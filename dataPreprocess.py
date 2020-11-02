import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


image_size = (160, 160)
num_classes = 4
batch_size = 32


def prepare_path(input_dir, target_dir):
    """ Prepare paths of input images and target segmentation masks """
    file_list = []
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.jpg'):
            file_list.append(os.path.join(input_dir, file_name))

    input_image_paths = sorted(file_list)

    file_list.clear()

    for file_name in os.listdir(target_dir):
        if file_name.endswith('.png') and not file_name.startswith('.'):
            file_list.append(os.path.join(target_dir, file_name))

    target_image_paths = sorted(file_list)

    return input_image_paths, target_image_paths


def make_validation(input_image_paths, target_image_paths):
    """ Split our img paths into a training and a validation set """

    validation_samples = 1000
    random.Random(2020).shuffle(input_image_paths)
    random.Random(2020).shuffle(target_image_paths)

    train_input_image_paths = input_image_paths[:-validation_samples]
    train_target_image_paths = target_image_paths[:-validation_samples]

    validate_input_image_paths = input_image_paths[-validation_samples:]
    validate_target_image_paths = target_image_paths[-validation_samples:]

    return train_input_image_paths, train_target_image_paths, validate_input_image_paths, validate_target_image_paths


class Oxford(tf.keras.utils.Sequence):
    """ Prepare Sequence class to load and vectorized batches of data """

    def __init__(self, bat_size, img_size, input_img_path, target_img_path):
        self.bat_size = bat_size
        self.img_size = img_size
        self.input_img_path = input_img_path
        self.target_img_path = target_img_path

    def __len__(self):
        return len(self.target_img_path) // self.bat_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.bat_size

        batch_input_img_path = self.input_img_path[i: i + self.bat_size]
        batch_target_img_path = self.target_img_path[i: i + self.bat_size]

        x = np.zeros(shape=(batch_size,) + self.img_size + (3,), dtype='float32')
        for j, path in enumerate(batch_input_img_path):
            img = load_img(path=path, target_size=self.img_size)
            # print(img_to_array(img).shape)
            x[j] = img

        y = np.zeros(shape=(batch_size,) + self.img_size + (1,), dtype='uint8')
        for j, path in enumerate(batch_target_img_path):
            img = load_img(path=path, target_size=self.img_size, color_mode='grayscale')
            # print(img_to_array(img).shape)
            y[j] = np.expand_dims(img, 2)

        return x, y
