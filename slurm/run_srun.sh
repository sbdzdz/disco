export SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY
srun --partition=a100-preemptable --nodes=1 --cpus-per-task=1 --ntasks=1 --mem=50G --gres=gpu:a100:1 --time=0-06:00 \
singularity exec --nv --bind $WORK $WORK/singularity/codis_latest.sif $HOME/codis/slurm/command.sh