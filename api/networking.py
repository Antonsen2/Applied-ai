import asyncio
import logging
from settings import get_encryption_key
from aescipher import AESCipher


AES = AESCipher(get_encryption_key())
HOST_FIREML = "fireml"
PORT_FIREML = 5000
CHUNK_SIZE = 1024

logger = logging.getLogger(__name__)


async def image_to_model(client_id: bytes, image) -> str:
    logger.debug(f"Making connection to: {HOST_FIREML}:{PORT_FIREML}")
    reader, writer = await asyncio.open_connection(HOST_FIREML, PORT_FIREML)
    # image header data
    checksum = f"{len(image)}".encode()
    header_data = AES.encrypt(checksum + b" " + client_id)
    header = header_data + b" " * (CHUNK_SIZE - len(header_data))
    logger.debug("Sending header data")
    writer.write(header)
    await writer.drain()
    logger.debug("Sending image data")
    writer.write(AES.encrypt(image))
    await writer.drain()
    writer.write_eof()
    logger.debug("Reading response")
    response = await reader.read(CHUNK_SIZE)
    checksum, client_id, msg = AES.decrypt(response.strip()).split()
    logger.debug(f"Got response: {msg.decode()}")
    return msg.decode()
