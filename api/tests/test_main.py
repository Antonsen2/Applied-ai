import sys
sys.path.append('../')

from fastapi.testclient import TestClient
from main import app
from fastapi import status
import wildfire_control
from PIL import Image
import io
from typing import List
from httpx import AsyncClient
import pytest
from starlette.datastructures import UploadFile
from typing import List

client=TestClient(app=app)

# def test_classify_image(path:str, files: dict):
#     response = client.post(path, files=files)
#     assert response.status_code == 200
#     return response


def test_hello_world():
    response = client.get("/")
    assert response.status_code == 200


def test_api_classify_image():
    #image = Image.new('RGB', (250, 250), (255, 0, 0))
    #fake_image_file = Image.open(r'C:\Code\Applied-ai\test-image\F_4.jpg')
    #fake_image_file = io.BytesIO()
    image1 = open('C:/Code/Applied-ai/test-image/F_4.jpg', "rb")
    image2 = open('C:/Code/Applied-ai/test-image/forest-1.jpg', "rb")
    #print(image)
    #fake_image_file = Image.fromarray(fake_image_file, mode='1')
    #print(type(fake_image_file))
    # fake_image.save(fake_image_file, 'JPEG')
    # fake_image_file.seek(0)
    #image_bytes = fake_image_file.read()
    #fake_image_file.close()
    #test_file = UploadFile(file=[image], filename="F_4.jpg")
    #response = test_classify_image("/api/classify", {"images": [test_file]})#{"coords": coords})
    response = client.post("api/classify", files={"images": [image1, image2] })
    # with open('C:/Code/Applied-ai/test-image/F_1.jpg', "wb") as f:
    #     response = client.post("/api/classify", files={"images": f ("filename", f, "image/jpeg")})
    # async with AsyncClient(app=app, base_url="http://127.0.0.1:8000/") as ac:
    #     response = await ac.post("/api/classify", files={"images": [test_file]})
    #print([test_file])
    #print(test_file)
    print(response.text)
    print(response.json)
    assert response.status_code == 200
#     # return response
#     # print([[wildfire_control.run_model(wildfire_control.process_single_image(test_image)), "test.jpg"]],test_image, test_file)#, coords)
#     # assert response.json() == {"label": [[wildfire_control.run_model(test_image), "test.jpg"]], "image": test_file}#, "coords": coords}
#     return True

# def test_api_classify_image():
#     fake_image = Image.new('RGB', (250, 250), (255, 0, 0))
#     fake_image = bytearray(fake_image)
#     test_image = wildfire_control.process_single_image(fake_image)
#     # test_image = bytes('fake_image_bytes', 'utf-8')
#     # #test_image = test_image.read()
#     # test_image = wildfire_control.process_single_image(test_image)
#     test_file = UploadFile(file=test_image, filename='test.jpg')
#     coords = [[1,1],[2,2]]
#     print([[wildfire_control.run_model(wildfire_control.process_single_image(test_image)), "test.jpg"]],test_image, test_file, coords)
#     response = test_classify_image("/api/classify", {"images": [test_file]}, {"coords": coords})
#     assert response.json() == [{"result": wildfire_control.run_model(test_image), "filename": "test.jpg", "coords": coords}]