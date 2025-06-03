#!/bin/sh
# ./install.sh
## Install required python packages and downloads resrouces

# install packages
pip install -r ./requirements.txt

# download spacy language supports
python -m spacy download en
python -m spacy download it
python -m spacy download es
python -m spacy download fr
python -m spacy download pt
