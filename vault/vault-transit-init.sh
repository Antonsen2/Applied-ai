#! /bin/sh
CHECK_INTERVAL_S=1

if [ -z $VAULT_ADDR ]; then
  export VAULT_ADDR="http://127.0.0.1:8200"
fi

while $true; do
    sleep $CHECK_INTERVAL_S

    vault status

    if [ $? -eq 0 ]; then
        break
    fi

done

# login with root token at $VAULT_ADDR
vault login $VAULT_TOKEN

# enable vault transit engine
vault secrets enable transit

# create key fire with type aes256-gcm96
vault write -f transit/keys/fire type=aes256-gcm96
