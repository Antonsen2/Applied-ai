from fastapi import FastAPI, File
from fastapi.responses import HTMLResponse
import wildfire_control
import numpy as np
from keras_preprocessing.image import img_to_array, ImageDataGenerator
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import load_img
from PIL import Image
import io

"""
Using uvicorn to run API.
$pip install uvicorn
and then in the console type uvicorn main:app to see api.
"""

app = FastAPI()

@app.get('/')
def upload_button():

    html =  """
    <body>
    <h1>Image Classification</h1>
    <p>Select an image to classify:</p>
    <form id="upload-form" method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*">
      <button type="submit">Upload</button>
    </form>
    """

    return HTMLResponse(html)

@app.post("/")
def classify_image(image: bytes = File(...)):
    # Loads the image uploaded and resizes it
    image = Image.open(io.BytesIO(image))
    image = image.resize((250, 250))
    image = img_to_array(image)
    image =  image.reshape((1,  image.shape[0],    image.shape[1],  image.shape[2]))
    image = preprocess_input(image)
    result = wildfire_control.run_model(image)
    return f"{result}"
