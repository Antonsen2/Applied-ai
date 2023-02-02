from enum import Enum
import json
import logging
from typing import List
import uuid
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from wildfire_control import generate_client_id, remove_client_id
from networking import image_to_model


app = FastAPI()
app.mount("/static", StaticFiles(directory="./static"), name="static")
templates = Jinja2Templates(directory="./templates")
# Initialize images store that persists user uploaded images
# to make it possible to render via html img tag.
image_store = dict()

class LoggingLevel(Enum):
    NOTSET = logging.NOTSET
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    @classmethod
    def get_level(cls, level: str):
        return cls.__members__.get(level, cls.NOTSET).value

LOGGER_NAME = "main"
FORMAT = "| %(asctime)s | %(levelname)s | %(name)s | %(message)s |"
logging.basicConfig(level=logging.INFO, format=FORMAT)
LOGGER = logging.getLogger(LOGGER_NAME)


@app.get('/classify')
async def read_html(request: Request):
    return templates.TemplateResponse("classify.html", {"request": request})


@app.post("/classify")
async def classify_image(background_tasks: BackgroundTasks,
                         request: Request,
                         images: List[UploadFile] = File(...),
                         coords: List = None):
    client_id = generate_client_id()

    results = []
    for image in images:
        bytes_image = await image.read()

        # Predict image using model
        prediction = await image_to_model(client_id, bytes_image)
        label = json.dumps(prediction)

        # Save and prepare image path to be able to display on results page
        image_id = str(uuid.uuid1())
        image_store[image_id] = bytes_image

        # Add dict to results list to display on results page
        results.append({
            "filename": image.filename,
            "prediction": prediction,
            "label": label,
            "coords": coords,
            "image_path": f"/classify/result/image/{image_id}"
        })

        LOGGER.info("Prediction results for %s got label %s", image.filename, label)

    background_tasks.add_task(remove_client_id, client_id)

    return templates.TemplateResponse("classify_post.html", {
                                      "request": request,
                                      "results": results})


@app.get('/classify/result/image/{image_id}')
async def get_result_image(image_id):
    if image_id in image_store:
        return Response(content=image_store[image_id], media_type="image/jpg")
    else:
        raise HTTPException(status_code=404, detail="Item not found")


@app.post("/api/classify")
async def api_classify_image(background_tasks: BackgroundTasks,
                             images: List[UploadFile] = File(...),
                             coords: List = None):
    client_id = generate_client_id()

    prediction = []
    for image in images:
        bytes_image = await image.read()
        result = await image_to_model(client_id, bytes_image)

        if result:
            prediction.append({"result": result,
                               "filename": image.filename,
                               "coords": coords})
            LOGGER.info("Successfully predicted %s got result %s", image.filename,
                json.dumps(result))
        else:
            LOGGER.warning("Unable to predict %s got result %s", image.filename, result)

    json_prediction = json.dumps(prediction)

    background_tasks.add_task(remove_client_id, client_id)

    return JSONResponse(content=json_prediction)
