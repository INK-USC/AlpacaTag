# ALPACA
An Active Learning-based Crowd Annotation\\ Framework for Sequence Tagging 

MIT License

<https://github.com/chakki-works/doccano>
## Features

* Collaborative annotation
* Language independent
* Active Learning & Online learning

## Requirements

* Python 3.6+
* django 2.0.5+
* Google Chrome(highly recommended)
* Pytorch 0.3.1
```bash
pip install https://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
```
## Installation

**Setup Python environment**

First we need to install the dependencies. Run the following commands:

```bash
conda create -n alonea python==3.6
source activate alonea
pip install -r requirements.txt
cd ALONEA
```

Next we need to compile the frontend. Run the following commands:

```bash
cd server
npm install
npm run build
cd ..
```

Next we need to make migration. Run the following command:

```bash
python manage.py migrate
```

Next we need to create a user who can login to the admin site. Run the following command:


```bash
python manage.py createsuperuser
```

Enter your desired username and press enter.

```bash
Username: admin
```

You will then be prompted for your desired email address:

```bash
Email address: admin@example.com
```

The final step is to enter your password. You will be asked to enter your password twice, the second time as a confirmation of the first.

```bash
Password: **********
Password (again): *********
Superuser created successfully.
```

## Usage

### Start the development server


**Running Django development server**

```bash
python manage.py runserver
```
Now, open a Web browser and go to <http://127.0.0.1:8000>.
If you run on your own server, use this command.

```bash
python manage.py runserver 0.0.0.0:8000
```
