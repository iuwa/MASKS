
from PIL import Image
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import timeit
import os
# from tensorflow.keras import backend as K
from class_build_noise import *
import json


# Model configuration
path = "Models/"
project_path = "../Fruit-Images-Dataset/MASKS/"
data_dir = "../Fruit-Images-Dataset/Test/"
batch_size = 1
img_width, img_height, img_num_channels = 100, 100, 3
loss_function = sparse_categorical_crossentropy
no_classes = 10
no_epochs = 75
optimizer = Adam()
validation_split = 0.2
verbosity = 1
input_shape = (img_width, img_height, img_num_channels)

datagen = ImageDataGenerator()
test_it = datagen.flow_from_directory(
                            data_dir, 
                            batch_size=batch_size, 
                            target_size = (img_width, img_height), 
                            seed=123)
randInt = []
cc = 0
conflicts = {}
for dirr in os.listdir(path):
    model = keras.models.load_model(path+'/'+dirr)
    step_size = 1
    cpu_res = []
    results = np.array([])
    accu = 0
    nacu = 0
    for _class_name in os.listdir(data_dir):
        for _file in os.listdir(data_dir+_class_name):
            start_intervala = timeit.default_timer() 
            if _file.split(".")[-1] == "jpg":
                np_image = Image.open(data_dir+_class_name+"/"+_file)
                np_image = np.array(np_image).astype('float32')/255
                np_image = np.expand_dims(np_image, axis=0)
                neigh_man = NeighMan(dir_data=project_path, input_shape=input_shape, output_shape=np_image.shape)
                input_test_nm = neigh_man.get_neighborhood_manipulations_from_image(np_image)
                temp_pre = np.argmax(model.predict(np.vstack(input_test_nm)), axis=1)
                cpu_res += list(temp_pre)
                for _i in test_it.class_indices:
                    if temp_pre[0] == test_it.class_indices[_i]:
                        if _i == _class_name:
                            accu += 1
                        else:
                            if _i+"+"+_class_name in conflicts:
                                conflicts[_i+"+"+_class_name] += 1
                            else:
                                conflicts[_i+"+"+_class_name] = 1
                        break
                nacu += 1
    del model
    results = np.asarray(cpu_res)
    np.save(project_path+"Results/"+dirr+"-"+str(cc)+"-"+str(accu/nacu)+"-", results)# out_classes
    cc += 1

f = open("conflicts.json", "w+")
f.write(json.dumps(conflicts))
f.close()


