import numpy as np
import os
from tensorflow.keras.datasets import cifar10, fashion_mnist
import timeit
import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
(input_train, target_train), (input_test, target_test) = fashion_mnist.load_data()
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

path  = "Results"
agent_counter = 0

predictions = []
for j in range(len(target_test)):
    predictions.append([i for i in range(10)])
agent_counter = 0
results = []
start_interval = timeit.default_timer()
for dirr in os.listdir(path):
    np_array = np.load(path+"/"+dirr)
    agent_counter +=1 
    correct_answer = 0
    wrong_answer = 0
    conflict_answer = 0
    correct_assist = 0
    wrong_assist = 0
    for i in range(len(target_test)):
        predictions[i] = list(set(predictions[i]) & set(np_array[i]))
        if not len(predictions[i]):
            conflict_answer += 1
        elif len(predictions[i]) == 1:
            if predictions[i][0] == target_test[i]:
                correct_answer += 1
            else:
                wrong_answer += 1
                # plt.figure(figsize=(10,10))
                # plt.xticks([])
                # plt.yticks([])
                # plt.grid(False)
                # plt.imshow(input_test[i])
                # plt.xlabel("agent_counter-"+str(agent_counter)+"-instance-"+str(i)+"-"+class_names[predictions[i][0]]+"-but it is-"+class_names[target_test[i]])
                # plt.show()
        else:
            if target_test[i] in predictions[i]:
                correct_assist += 1
            else:
                wrong_assist += 1
    with open("Agents/"+str(agent_counter),"w") as f:
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
print(str(results).replace('},','},\n'))
with open("AgentPredictions.txt","w") as f:
    f.write(str(results))
