import os
import shutil
import pandas as pd
from Datasetutil.Torchdatasetutil import *
# path = './wildfire/train'

# df = pd.read_csv(f'{path}/_classes.csv')

# burned_images = df[df[' burned'] == 1]
# fire_and_smoke_images = df[df[' fireandsmoke'] == 1]
# normal = df[df[' normal'] == 1]

# images = [burned_images, fire_and_smoke_images, normal]

# print(len(burned_images), len(fire_and_smoke_images), len(normal))
# folder_name = ['burned', 'fireandsmoke', 'normal']
# try:
#     for fname in folder_name:
#         os.mkdir(f'{path}/{fname}')
# except OSError as error:
#     print(error)
# for i, folder in enumerate(folder_name):
#     for filename in images[i].iloc[:, 0]:
#         src = f'{path}/{filename}'
#         dst = f'{path}/{folder}'
#         shutil.copy(src, dst)


dataloaders, dataset_sizes, class_names, imageshape = loadimagefolderdataset(name='wildfire', path='./')

print(dataset_sizes)