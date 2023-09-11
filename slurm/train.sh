#!/bin/bash
#SBATCH --ntasks=1                                                                     # Number of tasks (see below)
#SBATCH --cpus-per-task=16                                                             # Number of CPU cores per task
#SBATCH --nodes=1                                                                      # Ensure that all cores are on one machine
#SBATCH --time=0-24:00                                                                 # Runtime in D-HH:MM
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

export PYTHONPATH=$PYTHONPATH:$HOME/codis

srun --gres=gpu:1 python $HOME/codis/codis/train.py wandb.group=continual_classification training.max_epochs=3 model=stn \
 dataset.tasks=300 dataset.batch_size=512 model.gamma=1.0 dataset.factor_resolution=10 dataset.shapes_per_task=10 dataset.train_split=98000 dataset.val_split=1000 dataset.test_split=1000