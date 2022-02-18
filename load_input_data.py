
import tensorflow as tf

# from tensorflow.keras import  layers, models
# import matplotlib.pyplot as plt

# from tensorflow.keras.datasets import cifar10, fashion_mnist, cifar100
# from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
# from tensorflow.keras.losses import sparse_categorical_crossentropy
# from tensorflow.keras.optimizers import Adam, SGD
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing import image_dataset_from_directory
# import matplotlib.pyplot as plt
from tensorflow import keras
# import random
# import timeit
# import numpy as np
# import pandas as pd
# import os

# FILES = "../"

train_df_path = "../Fruit-Images-Dataset/Training/"

def load_input_data(batch_size=128, image_size=(100,100)):
    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        validation_split=0.2,
        vertical_flip=True,
    )
    train_generator = img_gen.flow_from_directory(
        train_df_path,
        target_size=image_size,
        batch_size=batch_size,
        subset='training'
        )         
    validation_generator = img_gen.flow_from_directory(
        train_df_path, 
        target_size=image_size,
        batch_size=batch_size,
        subset='validation')  
    return train_generator, validation_generator

