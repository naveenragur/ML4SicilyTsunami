#!/bin/bash

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
source /home/${USER}/.bash_profile # loading bash_proile
conda activate /mnt/beegfs/nragu/tsunami/env # conda environment

# running commands
cd $MLDir/scripts/testing

# python checkgpu.py

# echo 'Running 01_preprocess.py'
# python 01_preprocess.py CT 6317 train 6317 #our training pick
# python 01_preprocess.py CT 6421 test 6317 #deform pick
# python 01_preprocess.py CT 2500 test 6317 #non deform pick

# echo 'Running jobs for experiments.py' #before INGV visit
# srun python main.py with 'reg=CT' 'train_size=3200' 'mask_size=3200' 'channels_on=[64,64]' #onshore
# srun python main.py with 'reg=CT' 'train_size=3200' 'mask_size=3200' 'channels_on=[64,32]' #onshore
# srun python main.py with 'reg=CT' 'train_size=3200' 'mask_size=3200' 'channels_on=[64,16]' #onshore
# srun python main.py with 'reg=CT' 'train_size=3200' 'mask_size=3200' 'channels_on=[64,8]' #onshore
# srun python main.py with 'reg=CT' 'train_size=2500' 'mask_size=6317' 'channels_on=[64,64]' #onshore
# srun python main.py with 'reg=CT' 'train_size=2500' 'mask_size=6317' 'channels_on=[128,128]' #onshore
# srun python main.py with 'reg=CT' 'train_size=2500' 'mask_size=6317' 'channels_on=[256,256]' #onshore
# srun python main.py with 'reg=CT' 'train_size=2500' 'mask_size=6317' 'channels_on=[256,256,256]' #onshore
# srun python main.py with 'reg=CT' 'train_size=2500' 'mask_size=6317' 'channels_on=[128,128,128]' #onshore
# srun python main.py with 'reg=CT' 'train_size=2500' 'mask_size=6317' 'channels_on=[64,64,64]' #onshore
# srun python main.py with 'reg=CT' 'train_size=6317' 'mask_size=6317' 'channels_on=[128,128,128]' #onshore

# echo 'Running jobs for testing experiments.py' #after INGV visit
python main.py with 'reg=CT' 'train_size=6317' 'mask_size=6317' 

# finished commands
echo ...........
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed seconds
