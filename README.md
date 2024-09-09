# Tsunami Onshore Hazard Prediction using Machine Learning

This Git project tracks the work related to the use of machine learning (ML) for tsunami onshore hazard prediction. The goal is to develop a surrogate model that can be linked with a regional offshore tsunami model, using offshore wave amplitude as a timeseries input.

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
   - Events are selected based on specific criteria, such as stratified sampling parameters(typically - magnitude, displacement, depth, location, source type, etc.)
   - In our work we are focusing on the following, and pick different sizes
     - Offshore wave amplitude at selected points (maximum, time of maximum, etc.)
     - Deformation characteristics (maximum, min, etc.)
     - Onshore inundation characteristics (maximum depth, area, etc.)

2. Splitting the Event Selection
   - The selected events are divided into training and testing sets(75:25).

3. Training the ML Model
   - The ML model is trained on the training set, with guidance based on the test set for hyperparameter tuning.
   - Pretraining an offshore encoder (using a large dataset, not just the limited training set)
   - Pretraining an deformation encoder (using a large dataset, not just the limited training set)
   - Training an onshore decoder using the training set(as full simulation data is limited)
   - Fine-tuning the decoder, interface, and encoder using the training set(using the limited full simulation data is limited)

4. Model Performance Evaluation
   - The performance of the model is assessed using the unused dataset:
     - Evaluation at control points to check misfit and bias in classification of flooding
     - Evaluation at all inundation locations (using a single goodness-of-fit metric) and for subsets of different types
     - Evaluation for events of specific magnitude,source, locations and tsunami parameters as maps and boxplots
     - Evaluation for results with different training approaches, training sizes

5. Model Application
   - The results are used to generate PTHA inundation maps for the regions of interest.
   - The results are compared with HPC based results for a full ptha eventset, subset considering events that cause local deformation and events that dont cause any local deformation.
