from os import path

from setuptools import setup, find_packages

# setup metainfo
libinfo_py = path.join('alpaca_serving_client', 'client', '__init__.py')
libinfo_content = open(libinfo_py, 'r').readlines()
version_line = [l.strip() for l in libinfo_content if l.startswith('__version__')][0]
exec(version_line)  # produce __version__

with open('requirements.txt') as f:
    require_packages = [line[:-1] if line[-1] == '\n' else line for line in f]

setup(
    name='alpaca-serving-client',
    version=__version__,  # noqa
    description='AlpacaTag - client side',
    url='https://github.com/INK-USC/AlpacaTag',
    author='Bill Yuchen Lin, Dong-Ho Lee',
    author_email='dongho.lee@usc.edu',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=require_packages,
    classifiers=(
        'Programming Language :: Python :: 3.6.5',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ),
    keywords='alpacatag nlp sequence tagging annotation framework serving',
)
