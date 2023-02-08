import json
import logging
import os
import uuid
from base64 import b64encode
from enum import Enum
from typing import List

from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from networking import image_to_classifier, image_to_detection
from wildfire_control import generate_client_id, remove_client_id

app = FastAPI()
app.mount("/static", StaticFiles(directory="./static"), name="static")
templates = Jinja2Templates(directory="./templates")

LOGGER_NAME = "main"
FORMAT = "| %(asctime)s | %(levelname)s | %(name)s | %(message)s |"
LOG_LEVEL = logging.getLevelName(os.getenv('LOG_LEVEL', 'INFO').upper())
logging.basicConfig(level=LOG_LEVEL, format=FORMAT)
LOGGER = logging.getLogger(LOGGER_NAME)


@app.get("/")
def home():
    return RedirectResponse("/home")


@app.get('/home')
async def read_html(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get('/classify')
async def read_html(request: Request):
    return templates.TemplateResponse("classify.html", {"request": request})


@app.post("/classify")
async def classify_image(background_tasks: BackgroundTasks,
                         request: Request,
                         images: List[UploadFile] = File(...),
                         coords: List = None,
                         dodetect: bool = Form(False)):
    client_id = generate_client_id()

    LOGGER.info("New Client %s created to classify image",
                client_id.decode("utf-8"))

    coords = coords[0] if len(coords[0]) > 0 else None

    results = []
    for image in images:
        bytes_image = await image.read()
        prediction = await image_to_classifier(client_id, bytes_image)

        detect_image = None
        if prediction == "fire" and dodetect:
            detect_image = await image_to_detection(client_id, bytes_image)

        if prediction == "fire":
            prediction = "Fire detected"
        else:
            prediction = "No fire detected"

        results.append({"filename": image.filename,
                        "filetype": image.content_type,
                        "obj_result": detect_image,
                        "prediction": prediction,
                        "image": b64encode(bytes_image).decode("utf-8")})

    background_tasks.add_task(remove_client_id, client_id)

    LOGGER.info("Client %s finsihed classify image", client_id.decode("utf-8"))

    return templates.TemplateResponse("classify_post.html", {
                                      "request": request,
                                      "results": results,
                                      "coords": coords})


@app.post("/api/classify")
async def api_classify_image(background_tasks: BackgroundTasks,
                             images: List[UploadFile] = File(...),
                             coords: List = None):
    client_id = generate_client_id()

    LOGGER.info("New Client %s created to api classify image",
                client_id.decode("utf-8"))

    prediction = []
    for image in images:
        bytes_image = await image.read()
        result = await image_to_classifier(client_id, bytes_image)
        prediction.append({"result": result,
                           "filename": image.filename,
                           "coords": coords})

    # TODO test with dict instead
    json_prediction = json.dumps(prediction)

    background_tasks.add_task(remove_client_id, client_id)

    LOGGER.info("Client %s finsihed api classify image",
                client_id.decode("utf-8"))

    return JSONResponse(content=json_prediction)
