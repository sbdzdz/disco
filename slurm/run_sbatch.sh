#!/bin/bash
#SBATCH --ntasks=1                                                                     # Number of tasks (see below)
#SBATCH --cpus-per-task=1                                                              # Number of CPU cores per task
#SBATCH --nodes=1                                                                      # Ensure that all cores are on one machine
#SBATCH --time=0-02:00                                                                 # Runtime in D-HH:MM
#SBATCH --gres=gpu:1                                                                   # Request 1 GPU
#SBATCH --mem=50G                                                                      # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/mnt/qb/work/bethge/dziadzio08/projects/codis/slurm/hostname_%j.out   # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/qb/work/bethge/dziadzio08/projects/codis/slurm/hostname_%j.err    # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=END,FAIL                                                           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sebastian.dziadzio@uni-tuebingen.de                                # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID

source $HOME/.bashrc
source $WORK/virtualenvs/codis/bin/activate

python -m pip install --user --upgrade pip setuptools
python -m pip install --user -r $HOME/codis/requirements.txt
python -m pip install --user -e $HOME/codis

python $HOME/codis/codis/train.py \
 --dsprites_path $WORK/datasets/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz \
 --wandb_dir $WORK/projects/codis/wandb --beta 1 --epochs 20 --batch_size 64 --log_every 200