#!/bin/bash
#options for sbatch
#SBATCH --account=PR_carisma
#SBATCH --job-name=oq_hdf5 # Job name
#SBATCH --time=0-24:00 # time limit (D-HH:MM)
#SBATCH --array=0-6
#SBATCH --ntasks=1 # total number of tasks (jobs) across all nodes (3 nodes x 10 jobs each)
#SBATCH --mem=256G # memory limit
#SBATCH --cpus-per-task=1 # number of CPU cores per task
#SBATCH --partition=cpuq
#SBATCH --nodelist=cn03
# #SBATCH -p gpuq # partition
# #SBATCH --gres=gpu:1 # number of GPUs
# #SBATCH --nodelist=gp02
#SBATCH --output ./log/run-%A_%a.txt       # Standard out goes to this file
#SBATCH --error ./log/error-%A_%a.txt      # Standard err goes to this file


# for calculating the amount of time the job takes
begin=`date +%s`
echo node: $HOSTNAME
echo start time: `date`
echo ...........

# loading modules or conda
source /home/${USER}/.bash_profile # loading bash_proi

# running commands
cd /mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/risk/hazard



sizes=(225 529 892 1658 3454 7071 true)
task_id=${SLURM_ARRAY_TASK_ID:-0}
size=${sizes[$task_id]}

if [[ "$size" != "true" ]]; then
  conda activate rioxarray_env
  python scripts/prep_depth_pred_stats.py \
    --sizes "$size" \
    --output-dir /mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/risk/hazard/npy_inputs
else
  conda activate rioxarray_env
  python scripts/prep_depth_true.py
fi

conda activate oq
export NUMBA_CACHE_DIR=/tmp/numba_cache_${USER}
export MPLCONFIGDIR=/tmp/matplotlib_${USER}
export XDG_CACHE_HOME=/tmp/xdg_cache_${USER}
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
python scripts/create_openquake_hdf5.py \
  --sizes "$size" \
  --npy-dir /mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/risk/hazard/npy_inputs \
  --oq-dir /mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/risk/hazard/oq_inputs

#quick commands
# cat /proc/cpuinfo 
# export CUDA_VISIBLE_DEVICES=3
# # python checkgpu.py
# nvidia-smi
# python prep_depth_true.py
# python scripts/prep_depth_pred.py 892
# python scripts/prep_depth_pred.py 1658
# python scripts/prep_depth_pred.py 3454
# python scripts/prep_depth_pred.py 7071

# finished commands
echo ...........
end=`date +%s`
elapsed=`expr $end - $begin`
echo Time taken: $elapsed seconds
