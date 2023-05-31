# Tsunami Onshore Hazard Prediction using Machine Learning

This Git project tracks the work related to the use of machine learning (ML) for tsunami onshore hazard prediction. The goal is to develop a surrogate model that can be linked with a regional offshore tsunami model, using offshore wave amplitude as a timeseries input.

## Simulation Data

The ML model is trained using simulation data provided by INGV for Eastern Sicily, with a focus on Catania and Siracusa. The dataset consists of 1212 events, and you can view the event details and data through the following HTML maps:

(Click the "Download Raw" button at the link to download the file, its html files created with folium)

- [Events Map Explorer](/resources/gis/html/map_events.html) 

Some predictions:
- [CT Event 93](/model/CT/plot/CTevent_example_93.html)
- [SR Event 12](/model/CT/plot/SRevent_example_12.html)


## Workflow

The workflow for this project is as follows:

0. Preprocessing and Data Analysis
   - Offshore statistics for all events and gauges
   - Onshore statistics for all events at both sites
   - Earthquake statistics for all events (already available)

1. Selection of Events for Experiment
   - Events are selected based on specific criteria, such as stratified sampling parameters:
     - Magnitude, displacement, depth, location, source type, etc.
     - Offshore wave amplitude at selected points (maximum, time of maximum, etc.)
     - Onshore inundation characteristics (maximum depth, area, etc.)

2. Splitting the Event Selection
   - The selected events are divided into training and testing sets.

3. Training the ML Model
   - The ML model is trained on the training set, with guidance based on the test set for hyperparameter tuning.
   - Pretraining an offshore encoder (ideally using the whole dataset, not just the training set)
   - Training an onshore decoder using the training set
   - Fine-tuning the decoder, interface, and encoder using the training set

4. Model Performance Evaluation
   - The performance of the model is assessed using the unused dataset:
     - Evaluation at control points
     - Evaluation at all inundation locations (using a single goodness-of-fit metric)
     - Evaluation for events of specific magnitude or return period levels
     - Evaluation for events of different types
