import os
from base64 import b64encode, b64decode
import hvac


class AESCipher:
    def __init__(self):
        hvac_client = dict(url=os.environ["VAULT_ADDR"],
                           token=os.environ["VAULT_TOKEN"])
        self.client = hvac.Client(**hvac_client)
        assert self.client.is_authenticated()

        self.encrypt_key = "fire"
        self.decrypt_key = self.encrypt_key

    def encrypt(self, encode_data: bytes):
        encode_base64 = b64encode(encode_data).decode("utf-8")
        cipher_data = self.client.secrets.transit.encrypt_data(
                name=self.encrypt_key,
                plaintext=encode_base64
        )
        data = cipher_data["data"]["ciphertext"].encode()
        return data

    def decrypt(self, encrypted_data: bytes):
        encrypted_data = encrypted_data.decode('utf-8')
        decrypt_data_response = self.client.secrets.transit.decrypt_data(
                name=self.decrypt_key,
                ciphertext=encrypted_data
        )
        return b64decode(decrypt_data_response["data"]["plaintext"])
