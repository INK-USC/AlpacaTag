FROM node:14.1.0-alpine as node-builder

WORKDIR /annotation

COPY annotation/AlpacaTag /annotation

RUN ls && pwd && npm install --no-cache && \
    npm run build





FROM python:3.6-slim-buster as final

WORKDIR /alpaca

COPY --from=node-builder /annotation /alpaca/annotation

COPY requirements.txt requirements.txt
COPY alpaca_client alpaca_client
COPY alpaca_server alpaca_server

RUN pip install --no-cache-dir \
    ./alpaca_client \
    ./alpaca_server \
    -r requirements.txt && \
    python -m spacy download en_core_web_sm

