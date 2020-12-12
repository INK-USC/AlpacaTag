# create superuser from env
echo 'import os; from django.contrib.auth import get_user_model; User = get_user_model(); User.objects.create_superuser(os.environ[\"ADMIN_USERNAME\"], os.environ[\"ADMIN_EMAIL\"], os.environ[\"ADMIN_PASSWORD\"])' | python manage.py shell

# start model serving
alpaca-serving-start &

# run annotation
python manage.py runserver 0.0.0.0:8000
