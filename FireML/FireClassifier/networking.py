import asyncio
import socket
import logging
from aescipher import AESCipher
from prediction import model_predict, preprocess_image


AES = AESCipher()
HOST_SERVER = socket.gethostname()
PORT_SERVER = 5000
CHUNK_SIZE = 1024

LOGGER_NAME = "networking"
LOGGER = logging.getLogger(LOGGER_NAME)

async def run_server():
    server = await asyncio.start_server(handler, HOST_SERVER, PORT_SERVER)
    LOGGER.info("Server running on %s:%d", HOST_SERVER, PORT_SERVER)

    async with server:
        await server.serve_forever()

    LOGGER.info("Server %s:%d closed", HOST_SERVER, PORT_SERVER)


async def handler(reader: asyncio.StreamReader,
                  writer: asyncio.StreamWriter) -> None:
    header = await reader.read(CHUNK_SIZE)
    checksum, client_id = AES.decrypt(header.strip()).split()
    checksum = int(checksum)
    client_id = client_id.decode()

    LOGGER.info("Receiving client %s image size %d", client_id, checksum)

    # recv image
    data = b''
    while not reader.at_eof():
        data += await reader.read(CHUNK_SIZE)

    # decrypt image
    data = AES.decrypt(data)

    # TODO verify checksum
    LOGGER.debug("client %s image received, expected: %d; got: %d", client_id,
                 checksum, len(data))

    # prepare image for model
    image = preprocess_image(data)
    LOGGER.debug("client %s image preprocess for model prediction", client_id)

    # use model
    LOGGER.debug("client %s starting image prediction", client_id)
    fire_prediction = model_predict(image)
    LOGGER.info("client %s image prediction %s", client_id, fire_prediction)

    # send back result
    client_id = client_id.encode()
    msg = client_id + b" " + fire_prediction.encode("utf-8")
    checksum = f"{len(msg)}".encode("utf-8")
    msg = checksum + b" " + msg

    LOGGER.debug("client %s sending response %s", client_id.decode(), msg.decode())

    msg = AES.encrypt(msg)
    response = msg + b" " * (CHUNK_SIZE - len(msg))

    writer.write(response)
    await writer.drain()

    writer.close()
    LOGGER.debug("Closed client %s socket communication", client_id.decode())
    await writer.wait_closed()
