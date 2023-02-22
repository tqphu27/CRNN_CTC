# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors,
# and stored with the default serving key
import tempfile
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
print('export_path = {}\n'.format(export_path))
model = load_model('/home/tima/CRNN-CTC/datasets/BIRTH/models/last_inference_model.h5')

tf.keras.models.save_model(
    model,
    '/home/tima/CRNN-CTC/models/5',
    overwrite=True,
    include_optimizer=False,
    save_format=None,
    signatures=None,
    options=None
)

print('\nSaved model:')
