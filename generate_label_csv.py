import csv
import glob
import os

import pandas as pd

image_dir = '/home/tima/tf-crnn-ctc/datasets/VietcomBank/img'
path = '/home/tima/tf-crnn-ctc/datasets/VietcomBank'
filenames = os.listdir(image_dir)
print(len(filenames))
# filenames = sorted(filenames)
# labels = [f.split('.')[0].split('_')[-1] for f in filenames]
#
labels = [" "] * len(filenames)

df = pd.DataFrame({
    "filename": filenames,
    "label": labels
})


df.to_csv(os.path.join(path,"label.csv"), index=False)

