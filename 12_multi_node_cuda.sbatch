#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --mem=20G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
module load cuda
# Activate conda environment
eval "$(conda shell.bash hook)"
# Change conda environment name, if necessary
conda activate cs336_systems
# Get a unique port for this job based on the job ID
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_PORT: ${MASTER_PORT}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
srun python cs336-systems/cs336_systems/3_2_dist_comm_multi_node.py
