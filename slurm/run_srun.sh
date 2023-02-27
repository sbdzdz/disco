export SINGULARITYENV_WANDB_API_KEY=$WANDB_API_KEY
srun --partition=gpu-2080ti-preemptable --nodes=1 --cpus-per-task=1 --ntasks=1 --mem=50G --gres=gpu:rtx2080ti:1 --time=0-06:00 \
singularity exec --nv --bind /mnt/qb/work/bethge/dziadzio08 $HOME/singularity/python_build_latest.sif $HOME/codis/slurm/command.sh