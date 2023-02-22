import glob
from PIL import Image

# Get all images
image_path = '/home/tima/CRNN-CTC/datasets/ID/data_id'
images = glob.glob(image_path + '/**png') 
h = 0
w = 0

# For all images open them with PIL and get the image size
for image in images:
    with Image.open(image) as im:
        width, height = im.size
        h += height
        w += width

print(h/len(images),w/len(images))