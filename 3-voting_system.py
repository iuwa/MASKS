import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

label_dir = "../Fruit-Images-Dataset/MASKS/"
data_dir = "../Fruit-Images-Dataset/Test/"
path  = "../Fruit-Images-Dataset/MASKS/"
Project_path  = "../Fruit-Images-Dataset/MASKS/"

datagen = ImageDataGenerator()
test_it = datagen.flow_from_directory(
                            data_dir, 
                            seed=123)

def load_labels():
    label_arr = []
    for _class_name in os.listdir(data_dir):
        for _ in os.listdir(data_dir+_class_name):
            label_arr.append(test_it.class_indices[_class_name])
    return label_arr
target_test = load_labels()
class_no = len(test_it.class_indices)
agent_counter = 0
predictions = []
for j in range(len(target_test)):
    predictions.append([i for i in range(class_no)])
agent_counter = 0
results = []
votes = []
for i in range(len(target_test)):
    votes.append({})
    for j in range(class_no):
        votes[i][j] = 0
for i in range(len(target_test)):
    votes.append({})
    for j in range(class_no):
        votes[i][j] = 0
for dirr in os.listdir(path+"Results"):
    np_array = np.load(path+"Results"+"/"+dirr)
    agent_counter +=1 
    correct_answer = 0
    wrong_answer = 0
    conflict_answer = 0
    correct_assist = 0
    wrong_assist = 0
    for i in range(len(target_test)):
        votes[i][np_array[i*51]] += 1
        max_vote_no = 0
        max_vote_index = []
        for j in range(class_no):
            if votes[i][j] > max_vote_no:
                max_vote_no = votes[i][j]
                max_vote_index = [j]
            elif votes[i][j] == max_vote_no:
                max_vote_index.append(j)
        if not len(max_vote_index):
            conflict_answer += 1
        elif len(max_vote_index) == 1:
            if max_vote_index[0] == target_test[i]:
                correct_answer += 1
            else:
                wrong_answer += 1
        else:
            if target_test[i] in max_vote_index:
                correct_assist += 1
            else:
                wrong_assist += 1
    results.append({
        "agent_counter" : agent_counter,
        "correct_answer" : correct_answer,
        "wrong_answer" : wrong_answer,
        "conflict_answer" : conflict_answer,
        "correct_assist" : correct_assist,
        "wrong_assist" : wrong_assist})
with open(Project_path+"VotingAgentPredictions.txt","w") as f:
    f.write(str(results))
vote_tresh = 0.99
for vote_tresh_10 in range(1, 5):
    vote_tresh = vote_tresh_10/100.0
    agent_counter = 0
    predictions = []
    for j in range(len(target_test)):
        predictions.append([i for i in range(class_no)])
    agent_counter = 0
    results = []
    votes = []
    for i in range(len(target_test)):
        votes.append({})
        for j in range(class_no):
            votes[i][j] = 0
    for dirr in os.listdir(path+"Results"):
        np_array = np.load(path+"Results"+"/"+dirr)
        agent_counter +=1 
        correct_answer = 0
        wrong_answer = 0
        conflict_answer = 0
        correct_assist = 0
        wrong_assist = 0
        for i in range(len(target_test)):
            votes[i][np_array[i*51]] += 1
            max_vote_no = 0
            max_vote_index = []
            for j in range(class_no):
                if votes[i][j] > max_vote_no:
                    max_vote_no = votes[i][j]
                    max_vote_index = [j]
                elif votes[i][j] == max_vote_no:
                    max_vote_index.append(j)
            if not len(max_vote_index):
                conflict_answer += 1
            elif len(max_vote_index) == 1 and votes[i][max_vote_index[0]]>=agent_counter*vote_tresh:
                if max_vote_index[0] == target_test[i]:
                    correct_answer += 1
                else:
                    wrong_answer += 1
            else:
                if target_test[i] in max_vote_index:
                    correct_assist += 1
                else:
                    wrong_assist += 1
        results.append({
            "agent_counter" : agent_counter,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
            "conflict_answer" : conflict_answer,
            "correct_assist" : correct_assist,
            "wrong_assist" : wrong_assist})
    with open(Project_path+str(vote_tresh)+"VotingAgentPredictions.txt","w") as f:
        f.write(str(results))
