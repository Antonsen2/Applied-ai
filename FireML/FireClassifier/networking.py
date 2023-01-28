import logging
import asyncio
import socket
from aescipher import AESCipher
from settings import get_encryption_key
from prediction import model_predict, preprocess_image

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

AES = AESCipher(get_encryption_key())
HOST_SERVER = socket.gethostname()
PORT_SERVER = 5000
CHUNK_SIZE = 1024

async def run_server():
    logger.debug(f'Starting server on {HOST_SERVER}:{PORT_SERVER}')
    server = await asyncio.start_server(handler, HOST_SERVER, PORT_SERVER)
    async with server:
        await server.serve_forever()

async def handler(reader: asyncio.StreamReader,
                  writer: asyncio.StreamWriter) -> None:
    init_data = await reader.read(CHUNK_SIZE)
    checksum, client_id = AES.decrypt(init_data.strip()).split()
    checksum = int(checksum)

    logger.debug(f'Received image from client {client_id.decode()}')

    # recv image
    data = b''
    while not reader.at_eof():
        data += await reader.read(CHUNK_SIZE)

    # decrypt image
    data = AES.decrypt(data)

    # Verify checksum
    if len(data) != checksum:
        logger.error(f'Checksum mismatch: expected {checksum}, got {len(data)}')
        return
    else:
        logger.debug(f'Checksum match: {checksum}')

    # prepare image for model
    image = preprocess_image(data)

    # use model
    logger.debug(f'Predicting with model')
    fire_prediction = model_predict(image)

    # send back result
    msg = client_id + b" " + fire_prediction.encode("utf-8")
    checksum = f"{len(msg)}".encode("utf-8")
    msg = checksum + b" " + msg
    encrypted_msg = AES.encrypt(msg)
    response = encrypted_msg + b" " * (CHUNK_SIZE - len(encrypted_msg))

    logger.debug(f'Sending response: {fire_prediction}')
    writer.write(response)
    await writer.drain()

    writer.close()
    await writer.wait_closed()
