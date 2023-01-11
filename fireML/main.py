import os
import asyncio
from Crypto import Random
from aescipher import AESCipher
from networking import run_server


def app():
    # GET hashicorp vault
    key = "evensteven"
    aes = AESCipher(key)

    loop = asyncio.new_event_loop()
    loop.run_until_complete(run_server(aes))


if __name__ == '__main__':
    app()
