import asyncio
import pickle
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
    header_data = AES.encrypt(checksum + b" " + client_id)
    header = header_data + b" " * (CHUNK_SIZE - len(header_data))

    writer.write(header)
    await writer.drain()

    writer.write(AES.encrypt(image))
    await writer.drain()
    writer.write_eof()

    response = await reader.read(CHUNK_SIZE)
    checksum, client_id = AES.decrypt(response.strip()).split()

    data = b''
    while not reader.at_eof():
        data += await reader.read(CHUNK_SIZE)

    data = AES.decrypt(data)
    predictions = pickle.loads(data)
    image = plot_prediction(image, predictions)
    return image
