import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import platform


class LoadTestData:
    data_dir = "/media/iuwa/New Volume/Projects/MASKS/CatDog/test/test1"
    target_size = (224, 224)
    def __init__(self, 
                data_dir="/media/iuwa/New Volume/Projects/MASKS/CatDog/test/test1", 
                target_size = (224, 224)):
        if platform.system() == "Windows":
            self.dir_data = 'F:/Projects/MASKS/Fruit-Images-Dataset/MASKS/'
        else:
            self.dir_data = '/media/iuwa/New Volume/Projects/MASKS/Fruit-Images-Dataset/MASKS/'
        self.data_dir = data_dir
        self.target_size = target_size
    def load_test_data(self, start=0, end=-1):
        test_data = []
        for image_path in os.listdir(self.data_dir)[start: end]:
            image = tf.keras.preprocessing.image.load_img(
                                    self.data_dir+"/"+image_path,
                                    target_size=self.target_size,
                                    interpolation="nearest")
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            # input_arr = np.array([input_arr])  # Convert single image to a batch.
            test_data.append(input_arr)
        # print(len(test_data))
        return np.asarray(test_data)