#!/bin/bash

#options for sbatch
#SBATCH --account=PR_carisma
#SBATCH --job-name=CT-couple # Job name
#SBATCH --time=0-09:00 # time limit (D-HH:MM)
# #SBATCH --nodes=1 # number of nodes
# #SBATCH --partition=cpuq
#SBATCH -p gpuq # partition
#SBATCH --nodelist=gp02
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --mem=128G # memory pool for all cores
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

# echo 'Running 01_preprocess.py'
# python 01_preprocess.py CT 6317 train 6317 #our training pick
# python 01_preprocess.py CT 6421 test 6317 #deform pick
# python 01_preprocess.py CT 0 test 6317 #non deform pick

# echo 'Running jobs for experiments.py'
# srun python main.py with 'reg=CT' 'train_size=6317' 'lr=0.005' #offshore
# srun python main.py with 'reg=CT' 'train_size=6317' 'step_size=300' 'channels_on = [128,128,128]' #onshore
# srun python main.py with 'reg=CT' 'train_size=6317' 'channels_deform=[16,32,64,128]' #deform

srun python main.py with 'reg=CT' 'train_size=6317' 'channels_on = [256,256,256]' 'channels_deform=[16,32,64,128]' 'interface_layers=1' 'tune_nlayers=2' 
#with concat of x,dz as z before interface and decoder whose all layers are now tunable and remapped correction of skip connection

# finished commands
echo ...........
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed seconds
