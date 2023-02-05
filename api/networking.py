import asyncio
import logging
from fastapi import HTTPException
from aescipher import AESCipher


AES = AESCipher()
HOST_FIREML = "fireml"
PORT_FIREML = 5000

LOGGER_NAME = "networking"
LOGGER = logging.getLogger(LOGGER_NAME)


async def image_to_model(client_id: bytes, image) -> str:
    reader, writer = await asyncio.open_connection(HOST_FIREML, PORT_FIREML)

    # image header data
    checksum = f"{len(image)}".encode()
    header = AES.encrypt(checksum + b" " + client_id) + b"\n"

    LOGGER.info("Sending for client %s image size %s", client_id.decode(),
                checksum.decode())

    writer.write(header)
    await writer.drain()

    writer.write(AES.encrypt(image) + b"\n")
    await writer.drain()

    response = await reader.readline()
    checksum, client_id, msg = AES.decrypt(response.strip()).split()

    if msg == b"incomplete":
        # Failed to upload file to FireClassifier, Attempt two file transfer
        LOGGER.info("Client %s failed to upload file to FireClassifier",
                    client_id.decode())

        LOGGER.debug("Client %s starting attempt two", client_id.decode())

        writer.write(AES.encrypt(image) + b"\n")
        await writer.drain()

        response = await reader.readline()
        checksum, client_id, msg = AES.decrypt(response.strip()).split()

        if msg == b"unsuccessful":
            LOGGER.debug("Client %s attempt two failed", client_id.decode())
            LOGGER.info("Client %s unsuccessful upload file to FireClassifier",
                        client_id.decode())
            raise HTTPException(status_code=500, detail="unsuccessful file transfer")

        LOGGER.debug("Client %s attempt two successful", client_id.decode())

    LOGGER.info("Client %s received prediction %s", client_id.decode(), msg.decode())

    return msg.decode()
