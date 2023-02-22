from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import json
import requests
from PIL import Image 
import cv2
from matplotlib import pyplot as plt
import glob
import tensorflow.keras.backend as K
import os
import config
from numpy import asarray


def preprocessing(img):
   img = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
   img = cv2.resize(img, (400, 50))
   img = np.expand_dims(img, axis=-1)
   img_ = img / 255.0 
   return img_

def post_process(imgs,
               model_name='ocr',
               host='localhost',
               port=8501,
               signature_name="serving_default"):

   imgs = np.expand_dims(imgs, axis=0)
      
   data = json.dumps({
      "signature_name": signature_name,
      "instances": imgs.tolist()
   })
   # print(data)

   headers = {"content-type": "application/json"}
   json_response = requests.post(
      'http://{}:{}/v1/models/{}:predict'.format(host, port, model_name),
      data=data,
      headers=headers
   )
   if json_response.status_code == 200:
      y_pred = np.array(json.loads(json_response.text)['predictions'])

      out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:,:15]

      out = ''.join([characters[x] if x >= 0 else '' for x in out[0]])

      return out
   else:
      return None

if __name__ == '__main__':      
   import glob, os
   cfg = os.path.join(f'./datasets/BIRTH/models/config.json')
   config.load_config(cfg)
   characters = config.DatasetConfig.charset
   print(characters)
   # parent_dir = "/home/tima/CRNN-CTC/datasets/BIRTH/BIRTHDAY/"
   # for image_dir in glob.glob(os.path.join(parent_dir, '*.jpg')):
   #    # print(file)
   image_dir = '/home/tima/CRNN-CTC/datasets/BIRTH/BIRTHDAY/00000b7f3f7f1f1f_0_17-05-1975.jpg'
      
   img_=preprocessing(image_dir)
   # print(asarray(img_))
   print(post_process(imgs=img_))