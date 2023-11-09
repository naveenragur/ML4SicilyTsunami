#!/bin/bash

# options for sbatch
#SBATCH --job-name=deform-CT-2500 # Job name
#SBATCH --time=0-09:00 # time limit (D-HH:MM)
#SBATCH -p gpuq # partition
#SBATCH --nodelist=gp01
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --mem=64G # memory pool for all cores
#SBATCH --output ./log/run-%j.txt       # Standard out goes to this file
#SBATCH --error ./log/error-%j.txt        # Standard err goes to this file


# for calculating the amount of time the job takes
begin=`date +%s`
echo node: $HOSTNAME
echo start time: `date`
echo ...........

# setting up variables if provided
region=$1
size=$2
mode=$3
masksize=$4
split=$5
#TODO: add channels, zdim, and other hyper parameters  like batch size, learning rate, epoch of pretrian etc

# echo 'Running on' $region 'with' $size 'events' $mode 'mode' $masksize 'masksize'

# loading modules or conda
source /home/${USER}/.bash_profile # loading bash_proi
conda activate /mnt/beegfs/nragu/tsunami/env # conda environment

# running commands
cd $MLDir/scripts/deform

# python checkgpu.py

echo 'Running 01_preprocess.py'
# python 01_preprocess.py CT 2500 train 2500
# python 01_preprocess.py CT 6182 test 2500

# echo 'Running jobs for experiments.py'
srun python main.py with 'reg=CT' 'train_size=2500' 'test_size=6182'


# finished commands
echo ...........
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed seconds
