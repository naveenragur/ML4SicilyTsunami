# Tsunami Onshore Hazard Prediction using Machine Learning

This git project tracks the work related to the use of machine learning (ML) for tsunami onshore hazard prediction. The goal is to develop a stochastic inundation emulator that can be linked with inputs from a regional offshore tsunami model, using offshore wave timeseries and local deformation as input.

<img src="/resources/plots/Emulators.png" alt="Model Training Approach" height="400">

## Simulation Data

The ML model is trained using simulation data provided by INGV and NGI for Eastern Sicily, with a focus on Catania and Siracusa. The dataset consists of 53550 events, and you can view the event details and data through the following HTML maps:

(Click the "Download Raw" button at the link to download the file, its html files created with folium)
- [Events Map Explorer](/resources/gis/html/map_events.html)

The dataset is archived at main Zenodo link: [https://doi.org/10.5281/zenodo.13738078](https://doi.org/10.5281/zenodo.13738078) with parts as below.
- https://doi.org/10.5281/zenodo.13738078 (Part1) - Training Dataset and Model Checkpoints
- https://doi.org/10.5281/zenodo.13741284 (Part2) - Testing Dataset

## Contents
### configs
- YAML files with information on the Python packages and requirements to run
### data
- **events** contains folders on the events used for training, testing
- **simu** raw simulation data is stored here
- **processed** numpy binary files processed for fast read and write during experiments
### model
- **CT/multifoldMC/** folder with model outputs and plots used for Catania test site
- **SR/multifoldMC/** folder with model outputs and plots used for Siracusa test site
### resources
- background information generated or used in the experiments for reference
### scripts
- **JGR** contains notebooks, scripts for ml experiments and figures discussed in the manuscript 
- **interactive** additional notebooks useful for other file processing and handling in the project

## Usage
<img src="/resources/plots/EDArchJGR.png" alt="Model Training Approach" height="400">
- The project uses python, with experiments tracked with sacred tool and neptune for tracking and logging machine learning runs.
- Create a conda env using yml files provided in **configs** folder
- Download processed simulation files available in zenodo link:
- Use notebooks and code available in **scripts/** to run experiments, workflow described below

## Workflow
The workflow for this project is as follows:

0. Preprocessing and Data Analysis
   - Offshore statistics for all events and gauges
   - Onshore statistics for all events at both sites
   - Earthquake statistics for all events (already available)

1. Selection of Events for Experiment
   - Events are selected based on specific criteria, with stratified sampling of event parameters(typically - magnitude, displacement, depth, location, source type, etc.)
   - In our work we are focusing on the following, and pick different sizes
     - Offshore wave amplitude at selected points (maximum, time of maximum, etc.)
     - Deformation characteristics (maximum, min, etc.)
     - Onshore inundation characteristics (maximum depth, area, etc.)

<img src="/resources/plots/samplingflowchart.png" alt="Sampling Approach" height="400">

2. Splitting the Event Selection
   - The selected events are divided into training and testing sets(75:25). In our ensemble learning mode, a cross validation approach is adopted with  4 folds providing a shuffle across these training and test subsets. 

<img src="/resources/plots/StochasticEmulation.png" alt="Stochastic Emulation" height="400">

3.    Training the ML Model(Stochastic version of the encoder - decoder neural network) and prediction
   - The ML model is trained on the training set, with guidance based on the test set for hyperparameter tuning.
   - Here 4 encoder-decoder models are trained on each fold subsets of the training data.
   - For the stochastic version each of the four fold model is used to generate 100 realisations for each event from the test set.
   - The final prediction are presented with the mean and uncertainty bounds (+-2sigma) calculated from the 400 sample values.


4. Model Performance Evaluation
   - The performance of the model is assessed using the unused dataset:
     - Evaluation at control points to check misfit and bias in classification of flooding
     - Evaluation across inundation locations (using a single goodness-of-fit metric) and for subsets of different types
     - Evaluation for events of specific magnitude, source, locations and tsunami parameters as maps and boxplots
     - Evaluation for results training sizes

5. Model Application
   - The results are used to generate PTHA inundation maps for the regions of interest.
   - The results are compared with HPC based results for a full ptha eventset, subset considering events that cause local deformation and events that dont cause any local deformation.
   - These results are used to benchmark the emulation hazard with different training sizes against HPC and Stratified Importance Sampling based results.

<img src="/resources/plots/HazMaps.png" alt="PTHA Inundation Maps(HPC vs ML)" height="800">

## Useful References and Projects
- Ragu Ramalingam 2025 - [Advancing nearshore and onshore tsunami hazard approximation with machine learning surrogates](https://doi.org/10.5194/nhess-25-1655-2025)
- Abbate 2025 - [Importance sampling of seismic tsunami sources with near-field emphasis for inundation PTHA: benchmarking with complete ensembles](https://doi.org/10.1093/gji/ggaf034)
- Storrøsten 2024 - [Machine Learning Emulation of High Resolution Inundation Maps, Geophysical Journal International](https://doi.org/10.1093/gji/ggae151)
- Gibbons 2020 - [Probabilistic Tsunami Hazard Analysis: High Performance Computing for Massive Scale Inundation Simulations](https://doi.org/10.3389/feart.2020.591549)
- Basili 2020 -[The Making of the NEAM Tsunami Hazard Model 2018 (NEAMTHM18)](https://doi.org/10.3389/feart.2020.616594)

## Useful Github Projects
- [Tsunami Waveform and Inundation Emulator with Uncertainity](https://github.com/naveenragur/tsunami-surrogates.git) - A ML project for approximating tsunami wave height time series nearshore and maximum inundation depth onshore for the Japan Tohoku region, developed in Python/Pytorch.
- [Tsunami Inundation Emulator](https://github.com/norwegian-geotechnical-institute/tsunami-inundation-emulator.git) - A project for tsunami inundation depth prediction using machine learning, developed in Julia/Flex.

