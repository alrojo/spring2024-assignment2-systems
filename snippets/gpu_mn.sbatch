#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=00:02:00
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu

# Activate conda environment
eval "$(conda shell.bash hook)"
module load cuda
# Change conda environment name, if necessary
conda activate cs336_systems

# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_PORT: ${MASTER_PORT}"
echo "MASTER_ADDR: ${MASTER_ADDR}"

# Execute command for each task
srun python gpu_mn.py
