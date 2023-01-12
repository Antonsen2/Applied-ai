from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
import wildfire_control
from typing import List
from fastapi.templating import Jinja2Templates
import json

"""
Using uvicorn to run API.
$pip install uvicorn
and then in the console type uvicorn main:app to see api.
"""

app = FastAPI()
templates = Jinja2Templates(directory='./templates')

@app.get('/classify')
async def read_html(request: Request):
  return templates.TemplateResponse('classify.html', {"request": request})


@app.post("/classify")
async def classify_image(request: Request, images: List[UploadFile] = File(...), coords: List = None):
  prediction = []
  for image in images:
    bytes_image = await image.read()
    processed_image = wildfire_control.process_single_image(bytes_image)
    result = wildfire_control.run_model(processed_image)
    prediction.append([result, image.filename])
  json_prediction = json.dumps(prediction)
  return templates.TemplateResponse('classify_post.html', {"request": request, "label": json_prediction, "image": image, "coords": coords})


@app.post("/api/classify")
async def api_classify_image(images: List[UploadFile] = File(...), coords: List = None):
  prediction = []
  for image in images:
    bytes_image = await image.read()
    processed_image = wildfire_control.process_single_image(bytes_image)
    result = wildfire_control.run_model(processed_image)
    prediction.append({"result": result, "filename": image.filename, "coords": coords})
  json_prediction = json.dumps(prediction)
  return JSONResponse(content=json_prediction)
