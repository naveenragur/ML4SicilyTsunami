# README 
Git project to track the work related to the use of ML for tsunami onshore hazard prediction as a surrogate model which can be linked with a regional offshore tsunami model that can be used as the input by providing offshore wave amplitude as timeseries.

Simulation data for training the ML model provided by INGV for Eastern Scicily with inundation focused on Catania and Siracusa for 1212 events,check below html maps for quick view of events and data:
/resources/gis/html/event_example.html
/resources/gis/html/SRmap_explore.html


Workflow:
0. Preprocessing and data analysis
(a) offshore statistics for all events, all gauges
(b) onshore statistics for all events, both sites
(c) earthquake statistics for all events already available

1. Selection of events for experiment based on selection criteria like stratified sampling parameters include
(a) Magnitude, displacement, depth, location, source type, etc.
(b) Offshore wave amplitude at selected points - max, time of max, etc.
(c) Onshore inundation characteristics - max depth, area, etc.

2. Split a given selection of events into training and testing sets

3. Train a ML model on the training set with guidance based on testset for hyperparameter tuning
(a) Pretraining a offshore encoder(ideally I would do this with the whole dataset, not just training set)
(b) Training a onshore decoder (with all the training set)
(c) Fine tuning the decoder + interface + encoder (with the training set)

4. Check the performance of the model on the unused dataset 
(a) At control points
(b) At all the inundation locations (single goodness of fit metric)- plot at eq location to visualise as a map
(c) As events of specific magnitude or return period levels
(d) As events of different  

 




