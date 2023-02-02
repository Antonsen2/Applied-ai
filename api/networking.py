import asyncio
import logging
from aescipher import AESCipher


AES = AESCipher()
HOST_FIREML = "fireml"
PORT_FIREML = 5000
CHUNK_SIZE = 1024

LOGGER_NAME = "networking"
LOGGER = logging.getLogger(LOGGER_NAME)


async def image_to_model(client_id: bytes, image) -> str:
    reader, writer = await asyncio.open_connection(HOST_FIREML, PORT_FIREML)

    # image header data
    checksum = f"{len(image)}".encode()
    header_data = AES.encrypt(checksum + b" " + client_id)
    header = header_data + b" " * (CHUNK_SIZE - len(header_data))

    LOGGER.info("Sending client %s image size %s", client_id, checksum)

    writer.write(header)
    await writer.drain()

    writer.write(AES.encrypt(image))
    await writer.drain()
    writer.write_eof()

    response = await reader.read(CHUNK_SIZE)
    checksum, client_id, msg = AES.decrypt(response.strip()).split()

    LOGGER.debug("Received client %s response %s", client_id.decode(), msg.decode())

    return msg.decode()
