
readme.txt - this file

folders -
    sacred_logs - sacred experiment logs and backed up of results
    sbatch_logs - slurm logs
    sampling - scripts used to sample training events for the experiments 

Jupyter notebook to analyze the results of the ML experiment for different sites and approaches -
    00_results_ML_CT_direct.ipynb - 
    00_results_ML_CT_nodeform.ipynb
    00_results_ML_CT_withpretrain.ipynb
    00_results_ML_SR_direct.ipynb
    00_results_ML_SR_nodeform.ipynb
    00_results_ML_SR_withpretrain.ipynb

Slurm launch scripts - 
    run.sbatchCT
    run.sbatchSR

Scripts to run before or after experiments -
    01_preprocess.py
    02_postprocess.py
    03_reprocess.py
    04_compile_depths.py
    05_calcPerfGrid.py
    06_compare_depths.py
    07_calcPTHA.py

Main scripts with model code and training experiments- 
    experiment.py - main experiment script with emulator design and neural network training
    main.py - train pretraining model
    main_test.py - test pretraining model
    train.py - train direct models 
    test.py - test direct models

Utility scripts - 
    checkgpu.py
