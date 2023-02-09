import asyncio
import pickle
import logging
from fastapi import HTTPException
from aescipher import AESCipher
from plotprediction import plot_prediction


HOST_FIREML_CLASSIFIER = "fireclassifier"
PORT_FIREML_CLASSIFIER = 5000

HOST_FIREML_DETECTION = "firedetection"
PORT_FIREML_DETECTION = 5001

AES = AESCipher()

LOGGER_NAME = "networking"
LOGGER = logging.getLogger(LOGGER_NAME)


async def image_to_classifier(client_id: bytes, image) -> str:
    reader, writer = await asyncio.open_connection(HOST_FIREML_CLASSIFIER,
                                                   PORT_FIREML_CLASSIFIER)
    """The network socket communication to communicate with Fireclassifier.

        Protocol
        --------
        The communication follows by:
        1. Send metadata communication about file size and client id.
        2. Sending file.
            2.1 If receiving file transfer message with incomplete a second
            attempt will begin.
            2.2 Sending file.
            2.3 If receiving from server file transfer was unsuccessful the
                communication will be closed for the server and the raising
                error HTTP status code 500 with unsuccessful file transfer.
        3. Receiving with client id and the fireclassifier image prediction.
    """
    # image header data
    checksum = f"{len(image)}".encode()
    header = AES.encrypt(checksum + b" " + client_id) + b"\n"

    LOGGER.info("Client %s sending image size %s to FireClassifier",
                client_id.decode("utf-8"), checksum.decode("utf-8"))

    writer.write(header)
    await writer.drain()

    writer.write(AES.encrypt(image) + b"\n")
    await writer.drain()

    response = await reader.readline()
    checksum, client_id, msg = AES.decrypt(response.strip()).split()

    if msg == b"incomplete":
        # Failed to upload file to FireClassifier, Attempt two file transfer
        LOGGER.info("Client %s failed to upload file to FireClassifier",
                    client_id.decode("utf-8"))

        LOGGER.debug("Client %s starting attempt two to FireClassifier",
                     client_id.decode("utf-8"))

        writer.write(AES.encrypt(image) + b"\n")
        await writer.drain()

        response = await reader.readline()
        checksum, client_id, msg = AES.decrypt(response.strip()).split()

        if msg == b"unsuccessful":
            LOGGER.debug("Client %s attempt two failed to FireClassifier",
                         client_id.decode("utf-8"))

            LOGGER.info("Client %s unsuccessful upload file to FireClassifier",
                        client_id.decode("utf-8"))

            raise HTTPException(status_code=500,
                                detail="unsuccessful file transfer")

        LOGGER.debug("Client %s attempt two to FireClassifier successful",
                     client_id.decode("utf-8"))

    LOGGER.info("Client %s received prediction %s", client_id.decode("utf-8"),
                msg.decode("utf-8"))

    return msg.decode()


async def image_to_detection(client_id: bytes, image):
    reader, writer = await asyncio.open_connection(HOST_FIREML_DETECTION,
                                                   PORT_FIREML_DETECTION)
    """The network socket communication handler between client and server.

        Protocol
        --------
        The communication follows by:
        1. Metadata communication about file size and client id.
        2. Sending file.
            2.1 If receiving file transfer message with incomplete a second
            attempt will begin.
            2.2 Sending file.
            2.3 If receiving from server file transfer was unsuccessful the
                communction will be closed for the server and the raising
                error HTTP status code 500 with unsuccessful file transfer.
        3. Receiving with client id and detection models arrays prediction.
    """
    # image header data
    checksum = f"{len(image)}".encode("utf-8")
    header = AES.encrypt(checksum + b" " + client_id)

    LOGGER.info("Client %s sending image size %s to FireDetection",
                client_id.decode("utf-8"), checksum.decode("utf-8"))

    writer.write(header + b"\n")
    await writer.drain()

    writer.write(AES.encrypt(image) + b"\n")
    await writer.drain()

    response = await reader.readline()
    checksum, client_id, msg = AES.decrypt(response.strip()).split()

    if msg == b"incomplete":
        # FAILED to upload to FireClassifier, Attempt two file transfer
        LOGGER.info("Client %s failed to upload file to FireDetection",
                    client_id.decode("utf-8"))

        LOGGER.debug("Client %s starting attempt two to FireDetection",
                     client_id.decode("utf-8"))

        writer.write(AES.encrypt(image) + b"\n")
        await writer.drain()

        response = await reader.readline()
        checksum, client_id, msg = AES.decrypt(response.strip()).split()

        if msg == b"unsuccessful":
            LOGGER.debug("Client %s attempt two failed to FireDetection",
                         client_id.decode("utf-8"))

            LOGGER.info("Client %s unsuccessful upload file to FireDetection",
                        client_id.decode("utf-8"))

            raise HTTPException(status_code=500,
                                detail="unsuccessful file transfer")

        LOGGER.debug("Client %s attempt two to FireDetection successful",
                     client_id.decode("utf-8"))

    data = await reader.readline()
    data = AES.decrypt(data)

    predictions = pickle.loads(data)

    LOGGER.info("Client %s received prediction for FireDetection",
                client_id.decode("utf-8"))

    image = plot_prediction(image, predictions)

    LOGGER.debug("Client %s prediction Detection plotted out",
                 client_id.decode("utf-8"))

    return image
