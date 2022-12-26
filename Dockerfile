FROM python:3.6-slim-buster as final

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential\
    gcc \
    cmake \
    git \
    htop \
    tmux \
    ncdu \
    # cleanup:
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /alpaca

COPY requirements.txt requirements.txt
COPY alpaca_client/requirements.txt alpaca_client/requirements.txt
COPY alpaca_server/requirements.txt alpaca_server/requirements.txt

# 1 fetch small torch - for GPU support use a runtime image from hub.docker.com/r/pytorch/pytorch
# 2 install requirements for both django and model server
# 3 download spacy and distilbert data
RUN pip install --no-cache-dir \
    torch==1.7.1+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir \
    -r requirements.txt \
    -r alpaca_client/requirements.txt \
    -r alpaca_server/requirements.txt && \
    python -m spacy download en_core_web_sm && \
    python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("distilbert-multilingual-nli-stsb-quora-ranking")'




FROM node:14.1.0-alpine as node-builder

COPY annotation/AlpacaTag/server /annotation/server

WORKDIR /annotation/server

RUN npm install --no-cache && \
    npm run build




FROM final

WORKDIR /alpaca

COPY --from=node-builder /annotation/server /alpaca/annotation/server
COPY annotation annotation
COPY alpaca_client alpaca_client
COPY alpaca_server alpaca_server

RUN pip install --no-cache-dir \
    -e ./alpaca_client \
    -e ./alpaca_server

WORKDIR /alpaca/annotation/AlpacaTag

ENV ADMIN_USERNAME=admin
ENV ADMIN_PASSWORD=password
ENV ADMIN_EMAIL=admin@example.com

ENV DATABASE_URL=sqlite:////db.sqlite3
ENV ALLOW_SIGNUP=False
ENV DEBUG=True

# create django admin/migrations, start alpaca-serving, start django
COPY startup.sh startup.sh
EXPOSE 8000
ENTRYPOINT ["/bin/bash", "startup.sh"]
