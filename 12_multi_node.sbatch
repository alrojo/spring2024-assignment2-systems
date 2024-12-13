#!/bin/bash
#SBATCH --job-name=multi_dist
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=2
#SBATCH --mem=8G
#SBATCH --partition=batch
#SBATCH --account=mpsnyder
#SBATCH --time=01:00:00
# Activate conda environment
eval "$(conda shell.bash hook)"
# Change conda environment name, if necessary
conda activate cs336_systems
# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_PORT: ${MASTER_PORT}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
srun python cs336-systems/cs336_systems/dist_comm_multi_node.py
