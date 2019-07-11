from os import path

from setuptools import setup, find_packages

# setup metainfo
libinfo_py = path.join('alpaca_server', '__init__.py')
libinfo_content = open(libinfo_py, 'r').readlines()
version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][0]
exec(version_line)  # produce __version__

setup(
    name='alpaca_serving_server',
    version=__version__,
    description='alpacatag',
    url='https://github.com',
    author='Bill Yuchen Lin, Dong-Ho Lee',
    author_email='dongho.lee@usc.edu',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'numpy',
        'six',
        'pyzmq>=17.1.0',
        'GPUtil>=1.3.0',
        'termcolor>=1.1'
    ],
    extras_require={
        'http': ['flask', 'flask-compress', 'flask-cors', 'flask-json', 'bert-serving-client']
    },
    classifiers=(
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ),
    entry_points={
        'console_scripts': ['alpaca-serving-start=alpaca_server.server:main']
                            # 'bert-serving-benchmark=bert_serving.server.server:benchmark',
                            # 'bert-serving-terminate=bert_serving.server.server:terminate'],
    },
    keywords='alpacatag',
)