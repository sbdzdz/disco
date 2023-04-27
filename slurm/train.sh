#!/bin/bash
#SBATCH --ntasks=1                                                                     # Number of tasks (see below)
#SBATCH --cpus-per-task=16                                                             # Number of CPU cores per task
#SBATCH --nodes=1                                                                      # Ensure that all cores are on one machine
#SBATCH --time=0-12:00                                                                 # Runtime in D-HH:MM
#SBATCH --gres=gpu:1                                                                   # Request 1 GPU
#SBATCH --mem-per-cpu=16G                                                              # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/mnt/lustre/bethge/dziadzio08/projects/codis/slurm/hostname_%j.out    # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/lustre/bethge/dziadzio08/projects/codis/slurm/hostname_%j.err     # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=END,FAIL                                                           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sebastian.dziadzio@uni-tuebingen.de                                # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID

source $HOME/.bashrc
source $WORK/virtualenvs/codis/bin/activate

python -m pip install --upgrade pip setuptools
python -m pip install -r $HOME/codis/requirements.txt
python -m pip install -e $HOME/codis

python $HOME/codis/codis/train.py --wandb_dir $WORK/projects/codis/wandb --beta 1.5 --gamma 0.9 --tasks 6 --log_every_n_steps 10 --wandb_group continual_dsprites --max_epochs 50 --num_workers 16 --experiment codis --img_size 128 --batch_size 128