import sys
from fastapi.testclient import TestClient
from testing.main import app
import wildfire_control
from fastapi.templating import Jinja2Templates
import json


sys.path.append('../')
client = TestClient(app=app)


def test_api_classify_image():
    with open('test_data/F_4.jpg', 'rb') as f:
        image1 = f.read()

    with open('test_data/forest-1.jpg', 'rb') as f:
        image2 = f.read()

    response = client.post("api/classify", files=[("images", image1),
                           ("images", image2)])

    processed_image1 = wildfire_control.process_single_image(image1)
    processed_image2 = wildfire_control.process_single_image(image2)
    test_result = ({"result": wildfire_control.run_model(processed_image1),
                   "filename": "upload", "coords": None},
                   {"result": wildfire_control.run_model(processed_image2),
                   "filename": "upload", "coords": None})

    response = client.post("api/classify", files=[("images", image1),
                           ("images", image2)])
    assert response.status_code == 200
    assert response.json() == json.dumps(test_result)


def test_classify_image():
    with open('test_data/F_4.jpg', 'rb') as f:
        image1 = f.read()
    with open('test_data/forest-1.jpg', 'rb') as f:
        image2 = f.read()
    response = client.post("/classify", files=[("images", image1),
                           ("images", image2)])
    assert response.status_code == 200
