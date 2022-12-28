import pickle
import numpy as np
import os
import cv2

def load_image(image_path):
    image = cv2.imread(image_path)
    return [image]

def import_model(model_file):
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    return model

def run_model(image):
    model = import_model("C:\Code\Applied-ai\CNN50accuracy.sav")
    result = model.predict(image)
    return result
    
