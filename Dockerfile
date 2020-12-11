FROM node:14.1.0-alpine as node-builder

COPY annotation/AlpacaTag /annotation

WORKDIR /annotation

RUN npm install --no-cache && \
    npm run build




FROM python:3.7-slim-buster as final

COPY --from=node-builder /annotation /annotation

COPY alpaca_client /alpaca_client
COPY alpaca_server /alpaca_server

RUN pip install --no-cache-dir django ./alpaca_client ./alpaca_server

