#Readme for the codebase used for the 2025 paper submission on stochastic inundaiton emulator
#Date: 17/10/2025
#Author: Naveen Ragu Ramalingam (naveen.ragu@ngi.no)

#Main Workflow to be followed for tsunami emulation, we use slurm  for job scheduling using sbatch run.sbatchCT and run.sbatchSR for the two regions of interest (CT and SR).

1. Scripts 00_calcFlowdepth.py, 01_calcWaveheight.py and 02_calcWaveperiod.py are run to prepare some useful event statistics and summaries for selecting events for training.

2. Notebooks 03_sample_eve_CT.ipynb and 03_sample_eve_SR.ipynb are used to sample events from the INGV database. The events are selected based on the event statistics calculated in the previous step. The events list are then saved in /data/events/ as txt files.

3. Script 04_preprocess.py is run to prepare the data for machine learning emulation and saved in /data/processed/, a mask size derived from a set of sample events is used for maintaining homogenity in the locations of prediction.

4. Script train.py and test.py are used to train and test the machine learning models. The models are trained on the data prepared in the previous step. The models and predictions are saved in /models/<region>/<task>.

5. Script 05_compile_depths.py postprocesses the predicted mean and sigma depths into int format and save them as numpy arrays

6. Script 06_compare_depths.py plots depth predictions and errors for different models for a given event.

7. Script 07_calcPerfGrid.py calculates the performance metric at the prediction grids 

8. Script 08_calcPTHA.py CT Calculates the PTHA hazard curves at all prediction points for maps and subsets of events

9. Notebooks 09a_results_ML_CTerror.ipynb and 09b_results_ML_SRerror.ipynb are used collate results from emulation, make figures to evaluate prediction, compare hazard curves quickly, calculate posthoc uncertainity estimate, misfit at control points etc.

10. Directory plots/ contains notebooks for plotting figures used in the manuscript and supplements

#Other scripts used in the codebase but not run directly by the user
checkgpu.py #This script checks if a GPU is available and prints the name of the GPU if it is available.
experiment.py #This script is used to run the machine learning experiments. It contains all the functions need for data loading training, prediction and testing.
