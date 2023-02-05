import os
import logging
import asyncio
from fastapi import HTTPException
from aescipher import AESCipher


AES = AESCipher()
HOST_FIREML = "fireml"
PORT_FIREML = 5000


async def image_to_model(client_id: bytes, image) -> str:
    reader, writer = await asyncio.open_connection(HOST_FIREML, PORT_FIREML)

    # image header data
    checksum = f"{len(image)}".encode()
    header = AES.encrypt(checksum + b" " + client_id) + b"\n"

    writer.write(header)
    await writer.drain()

    writer.write(AES.encrypt(image) + b"\n")
    await writer.drain()

    response = await reader.readline()
    checksum, client_id, msg = AES.decrypt(response.strip()).split()

    if msg == b"incomplete":
        # Failed to upload file to FireClassifier, Attempt two file transfer
        writer.write(AES.encrypt(image) + b"\n")
        await writer.drain()

        response = await reader.readline()
        checksum, client_id, msg = AES.decrypt(response.strip()).split()

        if msg == b"unsuccessful":
            raise HTTPException(status_code=500, detail="unsuccessful file transfer")

    return msg.decode()
