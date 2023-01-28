import logging
import asyncio
from networking import run_server

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def app():
    # GET hashicorp vault
    loop = asyncio.new_event_loop()
    loop.run_until_complete(run_server())
    logger.debug("Server started")

if __name__ == '__main__':
    app()
