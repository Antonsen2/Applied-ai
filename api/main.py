"""
Using uvicorn to run API.
$pip install uvicorn
and then in the console type uvicorn main:app to see api.
"""
import os
from fastapi import FastAPI, File
from fastapi.responses import HTMLResponse
import numpy as np
from PIL import Image
import cv2
from wildfire_control import generate_client_id
from networking import image_to_model


app = FastAPI()

UPLOAD_FOLDER = "user_uploads"


@app.get('/')
def index():
    return "Hello world"


@app.get('/classify-image')
async def upload_button():
    html =  """
    <form id="upload-form" method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*">
      <button type="submit">Upload</button>
    </form>
    """
    return HTMLResponse(html)


@app.post("/classify-image")
async def classify_image(image: bytes = File(...)):
    # Loads the image uploaded and resizes it
    client_id = generate_client_id()

    prediction = await image_to_model(client_id, image)
    return { "prediction": prediction }
