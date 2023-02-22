import os
import numpy as np
import shutil
from os import path
# # Creating Train / Val / Test folders (One time use)
root_dir = '/home/tima/CRNN-CTC/datasets/all_id_4'
images = '/images'
labels = '/labels'

from glob import glob

excel_names = glob('/home/tima/CRNN-CTC/datasets/all_id_4/*.jpg')


os.makedirs(root_dir +'/train' + images)
os.makedirs(root_dir +'/train' + labels)
os.makedirs(root_dir +'/val' + images)
os.makedirs(root_dir +'/val' + labels)
os.makedirs(root_dir +'/test' + images)
os.makedirs(root_dir +'/test' + labels)

# Creating partitions of the data after shuffeling
currentCls = images
src = root_dir+currentCls # Folder to copy images from
print(currentCls)
allFileNames = []
for names in (excel_names):
   allFileNames.append(names)
   
np.random.shuffle(allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)*0.7), int(len(allFileNames)*0.85)])


train_FileNames = [name for name in train_FileNames.tolist()]
val_FileNames = [name for name in val_FileNames.tolist()]
test_FileNames = [name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))

# Copy-pasting images
for name in train_FileNames:
    if path.exists(name.replace("jpg","txt")):
        shutil.copy(name, "/home/tima/CRNN-CTC/datasets/all_id_4/train"+images)
        shutil.copy(name.replace("jpg","txt"), "/home/tima/CRNN-CTC/datasets/all_id_4/train"+labels)
    

for name in val_FileNames:
    if path.exists(name.replace("jpg","txt")):
        shutil.copy(name, "/home/tima/CRNN-CTC/datasets/all_id_4/val"+images)
        shutil.copy(name.replace("jpg","txt"), "/home/tima/CRNN-CTC/datasets/all_id_4/val"+labels)

for name in test_FileNames:
    if path.exists(name.replace("jpg","txt")):
        shutil.copy(name, "/home/tima/CRNN-CTC/datasets/all_id_4/test"+images)
        shutil.copy(name.replace("jpg","txt"), "/home/tima/CRNN-CTC/datasets/all_id_4/test"+labels)