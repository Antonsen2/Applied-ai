import sys
sys.path.append('../')

from fastapi.testclient import TestClient
from main import app
from fastapi import status, UploadFile
import wildfire_control

client=TestClient(app=app)

def test_classify_image(path:str, files: dict):
    response = client.post(path, files=files)
    assert response.status_code == 200
    return response


def test_api_classify_image():
    test_image = bytes('fake_image_bytes', 'utf-8')
    #test_image = test_image.read()
    test_image = wildfire_control.process_single_image(test_image)
    test_file = UploadFile(file=test_image, filename='test.jpg')
    #coords = [[1,1],[2,2]]
    response = test_classify_image("/classify", {"images": [test_file]}) #{"coords": coords})
    print([[wildfire_control.run_model(wildfire_control.process_single_image(test_image)), "test.jpg"]],test_image, test_file, coords)
    assert response.json() == {"label": [[wildfire_control.run_model(test_image), "test.jpg"]], "image": test_file, "coords": coords}

def test_api_classify_image():
    test_image = bytes('fake_image_bytes', 'utf-8')
    #test_image = test_image.read()
    test_image = wildfire_control.process_single_image(test_image)
    test_file = UploadFile(file=test_image, filename='test.jpg')
    coords = [[1,1],[2,2]]
    print([[wildfire_control.run_model(wildfire_control.process_single_image(test_image)), "test.jpg"]],test_image, test_file, coords)
    response = test_classify_image("/api/classify", {"images": [test_file]}, {"coords": coords})
    assert response.json() == [{"result": wildfire_control.run_model(test_image), "filename": "test.jpg", "coords": coords}]