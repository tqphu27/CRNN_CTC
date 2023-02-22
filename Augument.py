from PIL import Image 
import cv2
from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
import sys

import contextlib
import cv2
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import glob
from blurgenerator import motion_blur, gaussian_blur, lens_blur

image_dir = '/home/tima/CRNN-CTC/datasets/BIRTH/BIRTH/'

files = []
[files.extend(glob.glob(image_dir + '*.jpg'))]

# images = [cv2.imread(file) for file in files]

size_pixelated = [600,500,450,300,250,200,180,90,45,40]
print(image_dir)

for file in files[18000:20000]:
    # print(file)
    img = cv2.imread(file)
    img = cv2.resize(img, (400, 60)) 
    # img_pil = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # im_pil = Image.fromarray(img_pil)
    # size = (size_pixelated[5], size_pixelated[6])
    # image_tiny = im_pil.resize(size)    # resize it to a relatively tiny size
    # img = image_tiny.resize(im_pil.size,Image.NEAREST)
    # img = gaussian_blur(img, 1)
    # img = motion_blur(img, size=10, angle=10)
    # plt.imshow(img)
    # plt.show()
    # img.save(file)
    cv2.imwrite(file, img)
    print("complete")
    


