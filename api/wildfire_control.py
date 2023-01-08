import numpy as np
import cv2
import keras.models

def load_image(image_path):
    image = cv2.imread(image_path)
    return [image]

def import_model(model_file):
    return keras.models.load_model(model_file)

def run_model(image):
    model = import_model("C:\Code\Applied-ai\model_new(4).h5")
    pred = model.predict(image)
    pred = np.argmax(pred, axis=1)[0]
    labels = {0: 'Fire detected :)', 1: 'No fire detected :('}
    pred = labels[pred]
    return pred
    
