#!/usr/bin/env bash

# https://github.com/doccano/doccano/blob/v1.0.5/tools/run.sh
set -o errexit

# start model serving in background
echo "Starting alpaca-serving"
alpaca-serving-start &

sleep 3 && echo '\n\n'

echo "Running migrations on '$DATABASE_URL'"
python manage.py migrate

echo "Creating admin"
if [[ -n "${ADMIN_USERNAME}" ]] && [[ -n "${ADMIN_PASSWORD}" ]] && [[ -n "${ADMIN_EMAIL}" ]]; then
  DJANGO_SUPERUSER_PASSWORD=$ADMIN_PASSWORD python manage.py createsuperuser \
    --username "${ADMIN_USERNAME}" \
    --email "${ADMIN_EMAIL}" \
    --noinput \
  || true  # skip if already existing
fi

echo "Starting django"
python manage.py runserver "0.0.0.0:${PORT:-8000}"
