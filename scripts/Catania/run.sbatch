#!/bin/bash
#options for sbatch
#SBATCH --account=PR_carisma
#SBATCH --job-name=CT-ML # Job name
#SBATCH --time=0-12:00 # time limit (D-HH:MM)
#SBATCH --nodes=1 # number of nodes
# # SBATCH --partition=cpuq
#SBATCH -p gpuq # partition
#SBATCH --nodelist=gp01
#SBATCH --gres=gpu:1 # number of GPUs
#SBATCH --output ./sbatch_logs/run-%j.txt       # Standard out goes to this file
#SBATCH --error ./sbatch_logs/error-%j.txt        # Standard err goes to this file


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
cd $MLDir/scripts/Catania

# python checkgpu.py
# echo 'Running 01_preprocess.py' for deform full area
# python 01_preprocess.py CT 762 train 762 #min first set
# python 01_preprocess.py CT 1441 train 762 #second set
# python 01_preprocess.py CT 2589 train 762 #third set
# python 01_preprocess.py CT 4005 train 762 #fourth set
# python 01_preprocess.py CT 6734 train 762 #fifth set
# python 01_preprocess.py CT 0 test 762 #first test set
# python 01_preprocess.py CT 1 test 762 #second test set
# python 01_preprocess.py CT 2 test 762 #third test set
# python 01_preprocess.py CT 3 test 762 #fourth test set
# python 01_preprocess.py CT 53550 test 762 #fourth test set

# echo 'Running jobs for testing experiments.py' #after INGV visit
# python main.py with 'reg=CT' 'train_size=6734' 'mask_size=762' #offshore
# python main.py with 'reg=CT' 'train_size=6734' 'mask_size=762' #deform

# python train.py with 'reg=CT' 'train_size=762' 'mask_size=762' #onshore
# python test.py with 'reg=CT' 'train_size=762' 'mask_size=762' 'test_size=3' #evaluate
# python main.py with 'reg=CT' 'train_size=1441' 'mask_size=762' #onshore
# python main.py with 'reg=CT' 'train_size=2589' 'mask_size=762' 'lr_on=0.005' #onshore
# python main.py with 'reg=CT' 'train_size=4005' 'mask_size=762' #onshore
# python main.py with 'reg=CT' 'train_size=6734' 'mask_size=762' #onshore

# python test.py with 'reg=CT' 'train_size=4005' 'mask_size=762' 'test_size=0' #evaluate
# python test.py with 'reg=CT' 'train_size=4005' 'mask_size=762' 'test_size=1' #evaluate
# python test.py with 'reg=CT' 'train_size=4005' 'mask_size=762' 'test_size=2' #evaluate
# python test.py with 'reg=CT' 'train_size=4005' 'mask_size=762' 'test_size=3' #evaluate

python main.py with 'reg=CT' 'train_size=762' 'mask_size=762' 'test_size=3' #evaluate



# finished commands
echo ...........
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed seconds
