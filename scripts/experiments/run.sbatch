#!/bin/bash
# options for sbatch
#SBATCH --job-name=trainTsuML # Job name
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks-per-node=1 # number of tasks
#SBATCH --time=0-06:00 # time limit (D-HH:MM)
#SBATCH -p gpuq # partition
#SBATCH --gres=gpu:4 # number of GPUs
#SBATCH --mem=64G # memory pool for all cores
#SBATCH --output ./sbatch/run-%j.txt       # Standard out goes to this file
#SBATCH --error ./sbatch/error-%j.txt        # Standard err goes to this file

# for calculating the amount of time the job takes
begin=`date +%s`
echo node: $HOSTNAME
echo start time: `date`
echo ...........

# loading modules or conda
source /home/${USER}/.bash_profile # loading bash_proi
conda activate /mnt/beegfs/nragu/tsunami/env # conda environment

python main.py #with reg=SR #configs/special_config.json

# finished commands
echo ...........
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed seconds
