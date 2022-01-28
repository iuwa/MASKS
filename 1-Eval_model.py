import tensorflow as tf

from tensorflow.keras import  layers, models
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow import keras
import random
import timeit
import os
import numpy as np




start_interval_all = timeit.default_timer()
# Model configuration
path = "Models"
batch_size = 1500
img_width, img_height, img_num_channels = 28, 28, 1
loss_function = sparse_categorical_crossentropy
no_classes = 10
no_epochs = 75
optimizer = Adam()
validation_split = 0.2
verbosity = 1
input_shape=(img_width, img_height, img_num_channels)
all_salts = []
all_pepper = []
s_vs_p = 0.5
amount = 0.04
noise_counter = 100

(input_train, target_train), (input_test, target_test) = fashion_mnist.load_data()
# Normalize pixel values to be between 0 and 1
# input_train, input_test = input_train / 255.0, input_test / 255.0

# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_names = ['Top cover', 'Trouser', 'Shoes', 'Bag']
target_train = np.where(target_train==2, 0, target_train) 
target_train = np.where(target_train==3, 0, target_train) 
target_train = np.where(target_train==4, 0, target_train) 
target_train = np.where(target_train==6, 0, target_train) 
target_train = np.where(target_train==5, 2, target_train) 
target_train = np.where(target_train==7, 2, target_train) 
target_train = np.where(target_train==9, 2, target_train) 
target_train = np.where(target_train==8, 3, target_train) 
target_test = np.where(target_test==2, 0, target_test) 
target_test = np.where(target_test==3, 0, target_test) 
target_test = np.where(target_test==4, 0, target_test) 
target_test = np.where(target_test==6, 0, target_test) 
target_test = np.where(target_test==5, 2, target_test) 
target_test = np.where(target_test==7, 2, target_test) 
target_test = np.where(target_test==9, 2, target_test) 
target_test = np.where(target_test==8, 3, target_test) 
print(target_test[:100])


num_salt = np.ceil(amount * input_test[0].size * s_vs_p)
num_peper = np.ceil(amount* input_test[0].size * (1. - s_vs_p))
for i in range(noise_counter):
    all_salts.append([np.random.randint(0, i - 1, int(num_salt))  for i in input_test[0].shape])
    all_pepper.append([np.random.randint(0, i - 1, int(num_peper))  for i in input_test[0].shape])
fs = open("FashionNoiseSalt.txt", "w+")
fp = open("FashionNoisePeper.txt", "w+")
fs.write(str(all_salts))
fp.write(str(all_pepper))
fs.close()
fp.close()


def sp_noisy(image):
    out = []
    outTemp = np.copy(image)
    outTemp[all_salts[i]] = 1
    # Pepper mode
    outTemp[all_pepper[i]] = 0
    out.append(outTemp)
    return out


def get_neighborhood_manipulations(input_tests, noise_counter):
    manips = []
    counter = 0
    for input_test in input_tests:
        print(counter)
        manips+=[input_test]
        for _ in range(noise_counter):
            manips+=sp_noisy(input_test)
            counter += 1
    # return np.asarray(manips)
    return manips


stop_interval = timeit.default_timer() 
print("model Time Passed --------------------------------------------------------------------")
print(stop_interval-start_interval_all)
print("getting neighborhood")
input_test_nm = get_neighborhood_manipulations(input_test, noise_counter)
np_input_test_nm = np.asarray(input_test_nm)


stop_interval = timeit.default_timer() 
print("model Time Passed --------------------------------------------------------------------")
print(stop_interval-start_interval_all)
print("getting predictions")
randInt = []
cc = 0
for dirr in os.listdir(path):
    start_interval = timeit.default_timer()
    randInt = []
    model = keras.models.load_model(path+'/'+dirr)
    # score = model.evaluate(input_test, target_test, verbose=0)
    results = model.predict(np_input_test_nm)
    # result_array = [results]

    out_classes = np.reshape(np.argmax(results, axis=1), (-1, noise_counter+1))
    print(str(cc)+"-")
    stop_interval = timeit.default_timer() 
    print("all model Time Passed --------------------------------------------------------------------")
    print(stop_interval-start_interval)
    print("All Time Passed --------------------------------------------------------------------")
    print(stop_interval-start_interval_all)
    print("I --------------------------------------------------------------------")
    np.save("Results/"+str(cc)+"-"+dirr, out_classes)
    cc += 1
    del model


