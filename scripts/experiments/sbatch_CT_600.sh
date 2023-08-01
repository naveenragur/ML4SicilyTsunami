#!/bin/bash
#SBATCH --job-name=CT-600 # Job name
#SBATCH --nodes=1 # number of nodes
#SBATCH --ntasks-per-node=1 # number of tasks
#SBATCH --time=0-06:00 # time limit (D-HH:MM)
#SBATCH -p gpuq # partition
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --mem=64G # memory pool for all cores
#SBATCH --output ./sbatch/run-%j-CT-600.txt       # Standard out goes to this file
#SBATCH --error ./sbatch/error-%j-CT-600.txt      # Standard err goes to this file

begin=`date +%s`
echo node: $HOSTNAME
echo start time: `date`
echo ...........

source /home/${USER}/.bash_profile
conda activate /mnt/beegfs/nragu/tsunami/env
sleep 10
python main.py with 'reg=CT' 'train_size=600'

echo ...........
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed seconds
