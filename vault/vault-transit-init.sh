#! /bin/sh

VAULT_RETRIES=5

if [ -z $VAULT_ADDR ]; then
  export VAULT_ADDR="http://127.0.0.1:8200"
fi

until vault status > /dev/null 2>&1 || [ "$VAULT_RETRIES" -eq 0 ]; do
    echo "Waiting for vault to start...: $(($VAULT_RETRIES--))"
    sleep 1
done

# login with root token at $VAULT_ADDR
vault login $VAULT_TOKEN

# enable vault transit engine
vault secrets enable transit

# create key fire with type aes256-gcm96
vault write -f transit/keys/fire type=aes256-gcm96
