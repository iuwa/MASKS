
# from tensorflow.keras.layers import RandomRotation, RandomFlip, RandomZoom, Dense, Flatten, Conv2D, MaxPooling2D, Convolution2D, Activation, Dropout, GlobalAveragePooling2D
import random
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

input_shape = (100, 100, 3)
no_classes = 131

def define_model(randInt = [], 
                loss = "categorical_crossentropy", 
                optimizer = "adam", 
                metrics = "accuracy",
                no_classes = no_classes, 
                input_shape = input_shape):
    model = Sequential()
    layer_no = 11
    for _ in range(layer_no):
        randInt.append(random.randint(32,256))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(randInt[0], (3, 3), activation='relu'))
    model.add(layers.Conv2D(randInt[1], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(randInt[2], (3, 3), activation='relu'))
    model.add(layers.Conv2D(randInt[3], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(randInt[4], (3, 3), activation='relu'))
    model.add(layers.Conv2D(randInt[5], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(randInt[6], activation='relu'))
    model.add(layers.Dense(randInt[7], activation='relu'))
    model.add(layers.Dense(randInt[8], activation='relu'))
    model.add(layers.Dense(randInt[9], activation='relu'))
    model.add(layers.Dense(randInt[10], activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(no_classes, activation='softmax'))
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    return model, randInt