import random
import os
import shutil
from tqdm import tqdm

origin_data_dir = './Original_Large_Dataset'
save_dir = './New_Small_Data'
num = 0 #원하는 개수

files = os.listdir(origin_data_dir)

random_samples = random.sample(files, num)


for s in tqdm(random_samples):
    shutil.copy(origin_data_dir + s, save_dir + s)

