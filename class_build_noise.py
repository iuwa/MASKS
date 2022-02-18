import numpy as np
from class_load_test_data import *
import platform
print(platform.system())

class NeighMan:
    amount = 0.04
    noise_counter = 50 # 100
    all_salts = []
    all_pepper = []
    s_vs_p = 0.5
    noise_size = 224
    input_shape = (100, 100, 3)
    output_shape = (1, 100, 100, 3)
    dir_data = '../Fruit-Images-Dataset/MASKS/'
    
    def __init__(self, 
                amount=0.04, 
                noise_counter = 50, 
                s_vs_p = 0.5, 
                noise_size = 224, 
                input_shape = (100, 100, 3),
                output_shape = (1, 100, 100, 3),
                dir_data = '' ):
        if platform.system() == "Windows":
            self.dir_data = '../Fruit-Images-Dataset/MASKS/'
        else:
            self.dir_data = '../Fruit-Images-Dataset/MASKS/'
        self.amount = amount
        self.noise_counter = noise_counter
        self.s_vs_p = s_vs_p
        self.noise_size = noise_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.dir_data = dir_data
        if "all_salts_prime.npy" in os.listdir(self.dir_data):
            self.all_salts = np.load(self.dir_data+"all_salts_prime.npy")
        else:
            self.num_salt = np.ceil(self.amount * self.noise_size * self.s_vs_p)
            for _ in range(self.noise_counter):
                self.all_salts.append(np.asarray([np.random.randint(0, i-1, int(self.num_salt))  for i in self.input_shape]))
            np.save(self.dir_data+"all_salts_prime", np.asarray(self.all_salts))
            
        if "all_pepper_prime.npy" in os.listdir(self.dir_data):
            self.all_pepper = np.load(self.dir_data+"all_pepper_prime.npy")
        else:
            self.num_peper = np.ceil(self.amount* self.noise_size  * (1. - self.s_vs_p))
            for _ in range(self.noise_counter):
                self.all_pepper.append(np.asarray([np.random.randint(0, i-1, int(self.num_peper))  for i in self.input_shape]))
            np.save(self.dir_data+"all_pepper_prime", np.asarray(self.all_pepper))
            
    def sp_noisy(self, image):
        out = [image]
        for i in range(self.noise_counter):
            outTemp = np.copy(image).reshape(self.input_shape)
            outTemp[self.all_salts[i]] = 1
            outTemp[self.all_pepper[i]] = 0
            out.append(outTemp.reshape(self.output_shape))
        return out

    def get_neighborhood_manipulations_from_image(self, image):
        manips = []
        counter = 0
        manips+=self.sp_noisy(image)
        counter += 1
        return manips

    def get_neighborhood_manipulations(self, start=0, end=-1):
        manips = []
        counter = 0
        class_load_test_data = LoadTestData()
        input_test = class_load_test_data.load_test_data(start=start, end=end)
        for it in input_test:
            manips+=self.sp_noisy(it)
            counter += 1
        del class_load_test_data
        return manips