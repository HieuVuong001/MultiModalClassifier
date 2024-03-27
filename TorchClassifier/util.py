import os
import shutil
import pandas as pd
from Datasetutil.Torchdatasetutil import *
import random

base_path = './pokemon/train'


folder_name = ['Normal', 'Fire', 'Electric', 'Grass', 'Water', 'Fighting']


# 20% testing 80% training
for name in folder_name:
    directory = f'{base_path}/{name}'
    num_img = len(os.listdir(directory))
    all_files = os.listdir(directory)
    
    random.shuffle(all_files)

    # set out 20% of files for training
    test_length = int(0.2 * num_img)

    test_files = all_files[:test_length]
    train_files = all_files[test_length:]

    # move all test_file to a separate directory

    try:
        os.makedirs(f'./pokemon/val/{name}')
    except OSError as error:
        # acknoleged that folder already created
        print(error)

    for filename in test_files:
        src = f'./pokemon/train/{name}/{filename}'
        dst = f'./pokemon/val/{name}/{filename}'
        print(f'From : {src}')
        print(f'To : {dst}')
        shutil.move(src, dst)