#!/bin/bash
#SBATCH --ntasks=1                                                                 # Number of tasks (see below)
#SBATCH --cpus-per-task=1                                                          # Number of CPU cores per task
#SBATCH --nodes=1                                                                  # Ensure that all cores are on one machine
#SBATCH --time=0-06:00                                                             # Runtime in D-HH:MM
#SBATCH --gres=gpu:1                                                               # Request 1 GPU
#SBATCH --mem=50G                                                                  # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=/mnt/qb/work/bethge/dziadzio08/logs/slurm/codis/hostname_%j.out   # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/mnt/qb/work/bethge/dziadzio08/logs/slurm/codis/hostname_%j.err    # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=END,FAIL                                                       # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=sebastian.dziadzio@uni-tuebingen.de                            # Email to which notifications will be sent

# print info about current job
scontrol show job $SLURM_JOB_ID

# insert your commands here
export SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY
singularity exec --nv --bind $WORK $WORK/singularity/codis_latest.sif $HOME/codis/slurm/command.sh