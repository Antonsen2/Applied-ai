from fastapi import FastAPI, File, Request
from fastapi.responses import HTMLResponse
import wildfire_control
import numpy as np
from keras_preprocessing.image import img_to_array, ImageDataGenerator
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import load_img
from PIL import Image
import io
from starlette.responses import FileResponse 
from fastapi.templating import Jinja2Templates

"""
Using uvicorn to run API.
$pip install uvicorn
and then in the console type uvicorn main:app to see api.
"""

app = FastAPI()
templates = Jinja2Templates(directory='./templates')

@app.get('/')
async def read_html(request: Request):
  return templates.TemplateResponse('classify.html', {"request": request})
  #return FileResponse("templates/classify.html")


@app.post("/")
async def classify_image(request: Request, image: bytes = File(...)):
    # Loads the image uploaded and resizes it
    image = Image.open(io.BytesIO(image))
    image = image.resize((250, 250))
    image = img_to_array(image)
    image =  image.reshape((1,  image.shape[0],    image.shape[1],  image.shape[2]))
    image = preprocess_input(image)
    result = wildfire_control.run_model(image)
    return templates.TemplateResponse('classify_post.html', {"request": request, "label": result, "image": image})