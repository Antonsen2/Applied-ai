import logging
from uuid import uuid4

CLIENT_ID_POOL = []

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def generate_client_id() -> str:
    """Unique id generator uuid4 from RFC 4122"""
    client_id = str(uuid4()).encode()
    CLIENT_ID_POOL.append(client_id)
    logger.debug(f"Generated new client ID: {client_id}")
    return client_id


async def remove_client_id(client_id: bytes):
    CLIENT_ID_POOL.remove(client_id)
    logger.debug(f"Removed client ID: {client_id}")
