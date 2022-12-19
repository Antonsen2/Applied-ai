from fastapi import FastAPI, File
from fastapi.responses import HTMLResponse
import os

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


@app.get('/wildfire')
def upload_button():
    html =  """
    <form id="upload-form" method="post" enctype="multipart/form-data">
      <input type="file" name="image" accept="image/*">
      <button type="submit">Upload</button>
    </form>
    """
    return HTMLResponse(html)
    

@app.post('/wildfire')
def get_image(image: bytes = File(...)):
    # create the uploads folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    # save the image to the uploads folder
    with open(os.path.join(UPLOAD_FOLDER, "image.jpg"), "wb") as f:
        f.write(image)
    return {"filename": "image.jpg"}
