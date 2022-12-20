from fastapi import FastAPI, File
from fastapi.responses import HTMLResponse
import os
import wildfire_control
from PIL import Image
import numpy as np
import cv2

"""
Using uvicorn to run API.
$pip install uvicorn
and then in the console type uvicorn main:app to see api.
"""

app = FastAPI()

UPLOAD_FOLDER = "user_uploads"

@app.get('/')
def index():
    return "Hello world"


@app.get('/classify-image')
def upload_button():
    html =  """
    <form id="upload-form" method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*">
      <button type="submit">Upload</button>
    </form>
    """
    return HTMLResponse(html)
   
# @app.post('/wildfire')
# def get_image(image: bytes = File(...)):
#     # create the uploads folder if it doesn't exist
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)
#     # save the image to the uploads folder
#     with open(os.path.join(UPLOAD_FOLDER, "image.jpg"), "wb") as f:
#         f.write(image)
#     return "File uploaded successfully"


@app.post("/classify-image")
def classify_image(image: bytes = File(...)):
    # Preprocess the image
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, (250, 250))
    image = [image]
    image = np.array(image)
    print(image.shape)
    result = wildfire_control.run_model(image)
    return f"{result}"