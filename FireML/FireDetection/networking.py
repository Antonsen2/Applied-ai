import asyncio
import socket
import pickle
import logging
from aescipher import AESCipher
from prediction import preprocess_image, model_predict


AES = AESCipher()
HOST_SERVER = socket.gethostname()
PORT_SERVER = 5001

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
    header = await reader.readline()
    checksum, client_id = AES.decrypt(header.strip()).split()
    checksum = int(checksum)
    client_id = client_id.decode()

    LOGGER.info("Receiving client %s image size %d", client_id, checksum)

    file = await recv_file(reader)

    LOGGER.debug("client %s image received, expected: %d; got: %d", client_id,
                 checksum, len(file))

    if checksum != len(file):
        # FAILED transferring the file
        LOGGER.info("client %s incomplete file", client_id)
        msg = "incomplete"
        response = await package_response(client_id, msg)
        writer.write(response)
        await writer.drain()

        LOGGER.debug("client %s attempt 2 receiving file", client_id)

        file = await recv_file(reader)

        LOGGER.debug("client %s image received, expected: %d; got: %d",
                     client_id, checksum, len(file))

        if checksum != len(file):
            # FAILED transferring again, action close communication
            LOGGER.debug("client %s unsuccessful transfer", client_id)
            msg = "unsuccessful"
            response = await package_response(client_id, msg)
            writer.write(response)
            await writer.drain()

            writer.close()
            LOGGER.debug("Closed client %s socket communication", client_id)
            await writer.wait_closed()
            return None

    # prepare image for model
    image = preprocess_image(file)
    LOGGER.debug("client %s image preprocess for model prediction", client_id)

    # use model
    LOGGER.debug("client %s starting image prediction", client_id)
    fire_prediction = model_predict(image)
    LOGGER.info("client %s image object detection prediction finished",
                client_id)

    # SEND prediction header
    response = await package_response(client_id, "success")
    writer.write(response)
    await writer.drain()

    LOGGER.debug("client %s sent header response", client_id)

    # SEND prediction
    prediction_data = pickle.dumps(fire_prediction)
    writer.write(AES.encrypt(prediction_data) + b"\n")
    await writer.drain()

    LOGGER.info("client %s object detection sent", client_id)

    writer.close()
    LOGGER.debug("Closed client %s socket communication", client_id)
    await writer.wait_closed()


async def recv_file(reader: asyncio.StreamReader) -> bytes:
    data = await reader.readline()
    file = AES.decrypt(data)
    return file


async def package_response(client_id: str, msg: str) -> bytes:
    client_id, msg = client_id.encode("utf-8"), msg.encode("utf-8")
    msg = client_id + b" " + msg
    checksum = f"{len(msg)}".encode("utf-8")
    msg = checksum + b" " + msg

    LOGGER.debug("client %s sending response %s", client_id.decode(),
                 msg.decode())

    response = AES.encrypt(msg)
    return response + b"\n"
