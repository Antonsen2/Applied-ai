import logging
from uuid import uuid4


CLIENT_ID_POOL = []
LOGGER_NAME = "client id"
LOGGER = logging.getLogger(LOGGER_NAME)


def generate_client_id() -> bytes:
    """Unique id generator uuid4 from RFC 4122"""
    client_id = str(uuid4()).encode("utf-8")
    CLIENT_ID_POOL.append(client_id)
    LOGGER.debug("Client id %s created", client_id.decode("utf-8"))
    return client_id


async def remove_client_id(client_id: bytes):
    LOGGER.debug("Client id %s deleted", client_id.decode("utf-8"))
    CLIENT_ID_POOL.remove(client_id)
