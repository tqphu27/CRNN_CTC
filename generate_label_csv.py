import csv
import glob
import os

import pandas as pd

image_dir = '/home/tima/CRNN-CTC/datasets/ID/data_id'
filenames = os.listdir(image_dir)
print(len(filenames))
name = []
for image in filenames:
    name.append('img/'+str(image))
filenames = sorted(filenames)
# labels = [f.split('.')[0].split('_')[-1] for f in filenames]

# print(name)
labels = [" "] * len(filenames)

df = pd.DataFrame({
    "filename": filenames,
    "label": labels
})


df.to_csv(os.path.join("data2.csv"), index=False)

