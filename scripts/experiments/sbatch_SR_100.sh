#!/bin/bash
#SBATCH --job-name=SR-100 # Job name
#SBATCH --nodes=1 # number of nodes
##SBATCH --nodelist=gp02
#SBATCH --ntasks-per-node=1 # number of tasks
#SBATCH --time=0-06:00 # time limit (D-HH:MM)
#SBATCH -p gpuq # partition
##SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --mem=64G # memory pool for all cores
#SBATCH --output ./sbatch/run-%j-SR-100.txt       # Standard out goes to this file
#SBATCH --error ./sbatch/error-%j-SR-100.txt      # Standard err goes to this file

begin=`date +%s`
echo node: $HOSTNAME
echo start time: `date`
echo ...........

source /home/${USER}/.bash_profile
conda activate /mnt/beegfs/nragu/tsunami/env
python main.py with 'reg=SR' 'train_size=100' #'batch_size=100'

echo ...........
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed seconds
