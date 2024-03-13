#!/bin/bash

#options for sbatch
#SBATCH --account=PR_carisma
#SBATCH --job-name=CT-Test # Job name
#SBATCH --time=0-12:00 # time limit (D-HH:MM)
#SBATCH --nodes=1 # number of nodes
# # SBATCH --partition=cpuq
#SBATCH -p gpuq # partition
#SBATCH --nodelist=gp01 # node to be used
#SBATCH --gres=gpu:1 # number of GPUs
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
cd $MLDir/scripts/testing

# python checkgpu.py

# echo 'Running 01_preprocess.py' for deform full area
# python 01_preprocess.py CT 6317 train 6317 #our training pick
# python 01_preprocess.py CT 6421 test 6317 #deform pick
# python 01_preprocess.py CT 2500 test 6317 #non deform BS pick
# python 01_preprocess.py CT 0 test 6317 #non deform BS + PS pick
# python 01_preprocess.py CT 1609 test 6317 #deform training pick

# echo 'Running jobs for testing experiments.py' #after INGV visit
# python main.py with 'reg=CT' 'train_size=6317' 'mask_size=6317' 'lr=0.0035' #offshore loss function and lr
# python main.py with 'reg=CT' 'train_size=6317' 'mask_size=6317' 'batch_size_deform=100' #deform full
# python main.py with 'reg=CT' 'train_size=6317' 'mask_size=6317' 'channels_on=[16,128,128]' #onshore parts

python main.py with 'reg=CT' 'train_size=6317' 'mask_size=6317' #testing


# finished commands
echo ...........
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed seconds
