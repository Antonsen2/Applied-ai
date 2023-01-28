import io
import numpy as np
from PIL import Image
import keras.models
from keras_preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

MODEL_PATH = "model/model.h5"
MODEL = keras.models.load_model(MODEL_PATH)


def preprocess_image(data):
    logger.debug("Starting image preprocessing")
    image = Image.open(io.BytesIO(data))
    image = image.resize((250, 250))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    logger.debug("Completed image preprocessing")
    return preprocess_input(image)


def model_predict(image) -> str:
    logger.debug("Starting model prediction")
    pred = MODEL.predict(image)
    pred = np.argmax(pred, axis=1)[0]
    labels = {0: 'fire', 1: 'no-fire'}
    pred = labels[pred]
    logger.debug("Completed model prediction")
    return pred
