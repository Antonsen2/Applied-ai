import numpy as np
from keras_preprocessing.image import img_to_array, ImageDataGenerator
from keras.applications.mobilenet_v2 import preprocess_input
import keras.models
from PIL import Image
import io

def process_single_image(image: bytes):
    """Takes an image in bytes processes it to the right format for the model.

    
    Returns: image(250, 250, 3)
    """
    image = Image.open(io.BytesIO(image))
    image = image.resize((250, 250))
    image = img_to_array(image)
    image =  image.reshape((1,  image.shape[0],    image.shape[1],  image.shape[2]))
    image = preprocess_input(image)
    return image
    

def import_model(model_file):
    return keras.models.load_model(model_file)

def run_model(image):
    model = import_model("C:\Code\Applied-ai\model_new(4).h5")
    pred = model.predict(image)
    pred = np.argmax(pred, axis=1)[0]
    labels = {0: 'Fire detected', 1: 'No fire detected'}
    pred = labels[pred]
    return pred
    
