FROM python:3.6-slim-buster as final

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libicu-dev \
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
RUN pip install --no-cache-dir \
    -r requirements.txt \
    -r alpaca_client/requirements.txt \
    -r alpaca_server/requirements.txt && \
    python -m spacy download en_core_web_sm




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
ENV DEBUG=True

EXPOSE 8000
ENTRYPOINT ["/bin/bash", "-c", '\
    echo "import os; from django.contrib.auth import get_user_model; User = get_user_model(); User.objects.create_superuser(os.environ[\"ADMIN_USERNAME\"], os.environ[\"ADMIN_EMAIL\"], os.environ[\"ADMIN_PASSWORD\"])" | python manage.py shell && \
    alpaca-serving-start & \
    python manage.py runserver 0.0.0.0:8000']
