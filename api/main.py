import json
from typing import List
from fastapi import FastAPI, File, Request, UploadFile, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from wildfire_control import generate_client_id, remove_client_id
from networking import image_to_classifier, image_to_detection


app = FastAPI()
templates = Jinja2Templates(directory="./templates")


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

    results = []
    for image in images:
        bytes_image = await image.read()
        result = await image_to_classifier(client_id, bytes_image)

        detect_image = None
        if result == "fire" and dodetect:
            detect_image = await image_to_detection(client_id, bytes_image)

        results.append({"filename": image.filename,
                        "obj_result": detect_image,
                        "prediction": result,
                        "label": json.dumps(result),
                        "coords": coords})

    json_prediction = json.dumps(results)

    background_tasks.add_task(remove_client_id, client_id)

    return templates.TemplateResponse("classify_post.html", {
                                      "request": request,
                                      "label": json_prediction,
                                      "image": images,
                                      "results": results,
                                      "coords": coords})


@app.post("/api/classify")
async def api_classify_image(background_tasks: BackgroundTasks,
                             images: List[UploadFile] = File(...),
                             coords: List = None):
    client_id = generate_client_id()

    prediction = []
    for image in images:
        bytes_image = await image.read()
        result = await image_to_classifier(client_id, bytes_image)
        prediction.append({"result": result,
                           "filename": image.filename,
                           "coords": coords})

    json_prediction = json.dumps(prediction)

    background_tasks.add_task(remove_client_id, client_id)
    return JSONResponse(content=json_prediction)
