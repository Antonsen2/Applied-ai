import pickle
import numpy as np
import os
import cv2
import keras.models

def load_image(image_path):
    image = cv2.imread(image_path)
    return [image]

def import_model(model_file):
    return keras.models.load_model(model_file)

def run_model(image):
    model = import_model("C:\Code\Applied-ai\model(2).h5")
    pred = model.predict(image)
    pred = np.argmax(pred,axis=1)
    # labels = ({'fire': 0, 'no-fire': 1})
    # labels = dict((v,k) for k,v in labels.items())
    # pred = [labels[k] for k in pred]
    pred = [np.round(x) for x in pred]
    return pred
    
