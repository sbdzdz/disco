#!/bin/bash
python -m pip install --user --upgrade pip
python -m pip install --user --upgrade setuptools
python -m pip install --user -r $HOME/codis/requirements.txt
python -m pip install --user -e $HOME/codis
python $HOME/codis/codis/train.py --dsprites_path $WORK/datasets/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz --wandb_dir $WORK/logs/wandb --beta 1 --epochs 1 --batch_size 128