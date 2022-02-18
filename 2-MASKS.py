import numpy as np
import os
from tensorflow.keras.datasets import cifar10, fashion_mnist
import timeit
import json
# from load_label_data import *
import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

label_dir = "/media/iuwa/New Volume/Projects/MASKS/Fruit-Images-Dataset/MASKS/"
data_dir = "/media/iuwa/New Volume/Projects/MASKS/Fruit-Images-Dataset/Test/"


datagen = ImageDataGenerator()
test_it = datagen.flow_from_directory(
                            data_dir, 
                            seed=123)
print(test_it.class_indices)
print(len(test_it.class_indices))


class_no = len(test_it.class_indices)
nei_man_no = 51

def load_labels():
    label_arr = []
    for _class_name in os.listdir(data_dir):
        for _ in os.listdir(data_dir+_class_name):
            label_arr.append(test_it.class_indices[_class_name])
    return label_arr



target_test = load_labels()

path  = "/media/iuwa/New Volume/Projects/MASKS/Fruit-Images-Dataset/MASKS/"
agent_counter = 0

predictions = []
for j in range(len(target_test)):
    predictions.append([i for i in range(class_no)])
agent_counter = 0
results = []
start_interval = timeit.default_timer()
for dirr in os.listdir(path+"Results"):
    np_array = np.load(path+"Results"+"/"+dirr)
    # print(np_array)
    agent_counter +=1 
    correct_answer = 0
    wrong_answer = 0
    conflict_answer = 0
    correct_assist = 0
    wrong_assist = 0
    for i in range(len(target_test)):
        predictions[i] = list(set(predictions[i]) & set(np_array[i*nei_man_no:(i+1)*nei_man_no]))
        if not len(predictions[i]):
            conflict_answer += 1
        elif len(predictions[i]) == 1:
            if predictions[i][0] == target_test[i]:
                correct_answer += 1
            else:
                wrong_answer += 1
        else:
            if target_test[i] in predictions[i]:
                correct_assist += 1
            else:
                wrong_assist += 1
    with open(path+"Agents/"+str(agent_counter),"w") as f:
        f.write(str(predictions))
    
    results.append({
        "agent_counter" : agent_counter,
        "correct_answer" : correct_answer,
        "wrong_answer" : wrong_answer,
        "conflict_answer" : conflict_answer,
        "correct_assist" : correct_assist,
        "wrong_assist" : wrong_assist})
stop_interval = timeit.default_timer() 
print("all agent Time Passed --------------------------------------------------------------------")
print(stop_interval-start_interval)
print(str(results).replace("},", "},\n"))
with open("AgentPredictions.txt","w") as f:
    f.write(str(results))
