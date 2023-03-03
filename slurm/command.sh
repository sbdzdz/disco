#!/bin/bash
python -m pip install --user --upgrade pip
python -m pip install --user --upgrade setuptools
python -m pip install --user -r $HOME/codis/requirements.txt
python -m pip install --user -e $HOME/codis
python $HOME/codis/codis/train.py --dsprites_path /mnt/qb/work/bethge/dziadzio08/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz --beta 1