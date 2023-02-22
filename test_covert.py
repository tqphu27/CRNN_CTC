import numpy as np
from PIL import Image

# Define width and height
w, h = 50, 100

# Read file using numpy "fromfile()"
with open('data.bin', mode='rb') as f:
    d = np.fromfile(f,dtype=np.uint8,count=w*h).reshape(h,w)

# Make into PIL Image and save
PILimage = Image.fromarray(d)
PILimage.save('result.png')