FROM python:3.9.16-alpine3.17 as builder

WORKDIR /api

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


FROM python:3.9.16-alpine3.17

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN addgroup -S api && adduser -S wildfire -G api
USER wildfire

WORKDIR /home/wildfire/api
COPY --chown=wildfire:api . .

CMD ["python", "main.py"]
