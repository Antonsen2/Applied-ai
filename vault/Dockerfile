FROM vault:1.12.2

COPY vault-transit-init.sh /vault-transit-init.sh
RUN chmod +x /vault-transit-init.sh

ENTRYPOINT ["/bin/sh", "-c", "/vault-transit-init.sh"]
