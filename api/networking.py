import asyncio
import pickle
from fastapi import HTTPException
from aescipher import AESCipher
from plotprediction import plot_prediction


CHUNK_SIZE = 1024
HOST_FIREML_CLASSIFIER = "fireclassifier"
PORT_FIREML_CLASSIFIER = 5000

HOST_FIREML_DETECTION = "firedetection"
PORT_FIREML_DETECTION = 5001

AES = AESCipher()


async def image_to_classifier(client_id: bytes, image) -> str:
    reader, writer = await asyncio.open_connection(HOST_FIREML_CLASSIFIER,
                                                   PORT_FIREML_CLASSIFIER)

    # image header data
    checksum = f"{len(image)}".encode()
    header_data = AES.encrypt(checksum + b" " + client_id)
    header = header_data + b" " * (CHUNK_SIZE - len(header_data))

    writer.write(header)
    await writer.drain()

    writer.write(AES.encrypt(image))
    await writer.drain()
    writer.write_eof()

    response = await reader.read(CHUNK_SIZE)
    checksum, client_id, msg = AES.decrypt(response.strip()).split()

    return msg.decode()


async def image_to_detection(client_id: bytes, image):
    reader, writer = await asyncio.open_connection(HOST_FIREML_DETECTION,
                                                   PORT_FIREML_DETECTION)
    # image header data
    checksum = f"{len(image)}".encode()
    header = AES.encrypt(checksum + b" " + client_id)

    writer.write(header+ b"\n")
    await writer.drain()

    writer.write(AES.encrypt(image) + b"\n")
    await writer.drain()

    response = await reader.readline()
    checksum, client_id, msg = AES.decrypt(response.strip()).split()

    if msg == b"incomplete":
        # FAILED to upload to FireClassifier, Attempt two file transfer
        writer.write(AES.encrypt(image) + b"\n")
        await writer.drain()

        response = await reader.readline()
        checksum, client_id, msg = AES.decrypt(response.strip()).split()

        if msg == b"unsuccessful":
            raise HTTPException(status_code=500,
                                detail="unsuccessful file transfer")

    data = await reader.readline()
    data = AES.decrypt(data)

    predictions = pickle.loads(data)

    image = plot_prediction(image, predictions)
    return image
