#!/bin/bash
python -m pip install --upgrade pip
python -m pip install -r $HOME/codis/requirements.txt
python -m pip install -e $HOME/codis
python $HOME/codis/codis/train.py --dsprites_path /mnt/qb/work/bethge/dziadzio08/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz