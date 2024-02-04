#!/bin/bash
#SBATCH --ntasks=1                                                                     # Number of tasks (see below)
#SBATCH --cpus-per-task=8                                                              # Number of CPU cores per task
#SBATCH --nodes=1                                                                      # Ensure that all cores are on one machine
#SBATCH --time=3-00:00                                                                 # Runtime in D-HH:MM
#SBATCH --gres=gpu:1                                                                   # Request 1 GPU
#SBATCH --mem=50G                                                                      # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/mnt/qb/work/bethge/dziadzio08/projects/disco/slurm/hostname_%j.out   # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/qb/work/bethge/dziadzio08/projects/disco/slurm/hostname_%j.err    # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=END,FAIL                                                           # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sebastian.dziadzio@uni-tuebingen.de                                # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID

additional_args="$@"

source $HOME/.bashrc
source $WORK/virtualenvs/disco/bin/activate

python -m pip install --upgrade pip setuptools
python -m pip install -r $HOME/disco/requirements.txt
python -m pip install -e $HOME/disco

export PYTHONPATH=$PYTHONPATH:$HOME/disco
export WANDB__SERVICE_WAIT=300
export HYDRA_FULL_ERROR=1
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

python $HOME/disco/disco/train.py $additional_args