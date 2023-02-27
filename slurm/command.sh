#!/bin/bash
python -m pip install --upgrade pip
python -m pip install -r $HOME/codis/requirements.txt
python -m pip install -e $HOME/codis
python $HOME/codis/train.py