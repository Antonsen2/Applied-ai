from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
import plotly.graph_objects as go
from typing import List
from fastapi.templating import Jinja2Templates
import json
import sys
import matplotlib.pyplot as plt, mpld3
sys.path.append('../')
import wildfire_control
"""
Using uvicorn to run API.
$pip install uvicorn
and then in the console type uvicorn main:app to see api.
"""

app = FastAPI()
templates = Jinja2Templates(directory='../templates')


@app.get('/classify')
async def read_html(request: Request):
  return templates.TemplateResponse('classify.html', {"request": request})


@app.post("/classify")
async def classify_image(request: Request, images: List[UploadFile] = File(...), coords: List = None):
  results = []
  for image in images:
    bytes_image = await image.read()
    processed_image = wildfire_control.process_single_image(bytes_image)
    obj_result = wildfire_control.run_obj_model(bytes_image)
    prediction = wildfire_control.run_model(processed_image)
    results.append({
      "filename": image.filename,
      "obj_result": obj_result,
      "prediction": prediction,
      "label": json.dumps(prediction),
      "coords": coords, })
  return templates.TemplateResponse('classify_post.html', {"request": request, "results": results, "coords": coords})


@app.post("/api/classify")
async def api_classify_image(images: List[UploadFile]= File(...), coords: List = None):
  prediction = []
  print(images)
  for image in images:
    bytes_image = await image.read()
    processed_image = wildfire_control.process_single_image(bytes_image)
    
    result = wildfire_control.run_model(processed_image)
    prediction.append({"result": result, "filename": image.filename, "coords": coords, "image": image})
  json_prediction = json.dumps(prediction)
  
  return JSONResponse(content=json_prediction)
