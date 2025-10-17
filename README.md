# Tsunami Onshore Hazard Prediction using Machine Learning

This git project tracks the work related to the use of machine learning (ML) for tsunami onshore hazard prediction. The goal is to develop a stochastic inundation emulator that can be linked with inputs from a regional offshore tsunami model, using offshore wave timeseries and local deformation as input.

## Simulation Data

The ML model is trained using simulation data provided by INGV and NGI for Eastern Sicily, with a focus on Catania and Siracusa. The dataset consists of 53550 events, and you can view the event details and data through the following HTML maps:

(Click the "Download Raw" button at the link to download the file, its html files created with folium)
- [Events Map Explorer](/resources/gis/html/map_events.html) 

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

2. Splitting the Event Selection
   - The selected events are divided into training and testing sets(75:25). In our ensemble learning mode, a cross validation approach is adopted with  4 folds providing a shuffle across these training and test subsets. 

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
