import tensorflow as tf

from tensorflow.keras import  layers, models
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10, fashion_mnist, cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from tensorflow import keras
import random
import timeit
import numpy as np

# Model configuration
batch_size = 1500
loss_function = sparse_categorical_crossentropy
no_classes = 4
no_epochs = 15#75
optimizer = Adam()
validation_split = 0.2
verbosity = 1
model_no = 1000

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_names = ['Top cover', 'Trouser', 'Shoes', 'Bag']

(input_train, target_train), (input_test, target_test) = fashion_mnist.load_data()
print(input_test.shape)
if len(input_test.shape) > 3:
    img_width, img_height, img_num_channels = input_test.shape[1], input_test.shape[2], input_test.shape[3]
else:
    img_width, img_height, img_num_channels = input_test.shape[1], input_test.shape[2], 1
input_shape = (img_width, img_height, img_num_channels)
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')
input_train = input_train / 255
input_test = input_test / 255
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

def define_model(randInt, input_shape):
    model = models.Sequential()
    randInt.append(random.randint(64,256))
    randInt.append(random.randint(48,196))
    randInt.append(random.randint(32,128))
    randInt.append(random.randint(32,128))
    randInt.append(random.randint(24,96))
    randInt.append(random.randint(24,96))
    randInt.append(random.randint(16,64))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.15))
    model.add(layers.Conv2D(randInt[0], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.15))
    model.add(layers.Conv2D(randInt[1], (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.15))
    model.add(layers.Dense(randInt[2], activation='relu'))
    model.add(layers.Dropout(0.15))
    model.add(layers.Dense(randInt[3], activation='relu'))
    model.add(layers.Dropout(0.15))
    model.add(layers.Dense(randInt[4], activation='relu'))
    model.add(layers.Dropout(0.15))
    model.add(layers.Dense(randInt[5], activation='relu'))
    model.add(layers.Dropout(0.15))
    model.add(layers.Dense(randInt[6], activation='relu'))
    model.add(layers.Dropout(0.15))
    model.add(layers.Dense(no_classes, activation='softmax'))

    model.compile(loss=loss_function,
                optimizer=optimizer,
                metrics=['accuracy'])
    return model







start_interval_all = timeit.default_timer()
randInt = []
for i in range(model_no):
    start_interval = timeit.default_timer()
    randInt = []
    model = define_model(randInt, input_shape)
    

    history = model.fit(input_train, target_train,
                batch_size=batch_size,
                epochs=no_epochs,
                verbose=verbosity,
                validation_split=validation_split)
    score = model.evaluate(input_test, target_test, verbose=0)
    print("Test ====================================================================")
    print(f'Test score: {score}')
    print(model.metrics_names)
    print("Train ====================================================================")
    score_train = model.evaluate(input_train, target_train, verbose=0)
    print(f'Train score: {score_train}')
    print(model.metrics_names)
    print("End ====================================================================")
    tempStr = ""
    for j in randInt:
        tempStr += str(j) + "-"
    modelName = "st-"+str(score[1])+"s-"+str(score_train[1])+"-l-"+tempStr
    model.save("Models/"+modelName)    

    del model
    stop_interval = timeit.default_timer()  
    print("TIME -----------------------------------------------------------------")
    print(stop_interval-start_interval)
    print("Time Passed --------------------------------------------------------------------")
    print(stop_interval-start_interval_all)
    print("I --------------------------------------------------------------------")
