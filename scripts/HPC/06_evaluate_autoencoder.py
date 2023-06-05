import os
import sys
# import glob

import numpy as np
import model_utils as utils
from torchsummary import summary

try:
    MLDir = os.getenv('MLDir')
    SimDir = os.getenv('SimDir')
    reg = sys.argv[1] #CT or SR
    train_size = sys.argv[2] #eventset size for training
    mode = sys.argv[3] #train or test
    test_size = sys.argv[4] #eventset size for testing
except:
    raise Exception("*** Must first set environment variable")

#set seed
np.random.seed(0)

#TODO: could be read from a parameter file instead of hardcoding here
#feature dimensions for both regions and gauge numbers
if reg == 'SR':
    GaugeNo = list(range(53,58)) #rough pick for Siracusa
    x_dim = 1300  #lon
    y_dim = 948 #lat
    ts_dim = len(GaugeNo) #no of gauges time series
    control_points = [[37.01,15.29],
        [37.06757,15.28709],
        [37.05266,15.26536],
        [37.03211,15.28632]]
    

elif reg == 'CT':
    GaugeNo = list(range(35,44)) #for Catania
    x_dim = 912
    y_dim = 2224
    ts_dim = len(GaugeNo)
    control_points =  [[37.5022,15.0960],
        [37.48876,15.08936],
        [37.47193,15.07816],
        [37.46273,15.08527],
        [37.46252,15.08587],
        [37.45312,15.07874],
        [37.42821,15.08506],
        [37.40958,15.08075],
        [37.38595,15.08539],
        [37.35084,15.08575],
        [37.33049,15.07029],
        [37.40675,15.05037]
        ]

pts_dim = 480 #time steps

#load events data from simulation directory
event_list = os.listdir(f'{SimDir}')
flood_mask = ~np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{train_size}.npy')
nflood_grids = np.count_nonzero(flood_mask)
print(f'Number of flooded grids: {nflood_grids}')

n_eve = len(event_list)
print(f'Number of events to test: {n_eve}')

t_array = np.memmap(f'{MLDir}/data/processed/t_{reg}_{train_size}.dat',
                mode='r+',
                dtype=float,
                shape=(n_eve, ts_dim, pts_dim))

red_d_array = np.memmap(f'{MLDir}/data/processed/dflat_{reg}_{train_size}.dat',
                         mode='r+',
                         dtype=float,
                         shape=(n_eve, nflood_grids))

#TODO:split as decided by ratio which is default here, can be sys arg or even event list files
# input data
data_in = t_array
data_out = red_d_array

#directory to save model and outputs
if not os.path.exists(f'{MLDir}/model/{reg}/out/'):
    os.makedirs(f'{MLDir}/model/{reg}/out/')
if not os.path.exists(f'{MLDir}/model/{reg}/plot/'):
    os.makedirs(f'{MLDir}/model/{reg}/plot/')

#TODO: make these sys args as well, will be useful for hyperparameter tuning and loggin results aswell
#learning parameters
batch_size = 20
selected_epoch = None
#select a specific pretrained model #TODO: make this sys arg aswell
job = 'couple'

modeldef = [ts_dim,pts_dim,nflood_grids ] #save model definition to read for summary

utils.evaluateAE(data_in, #training data offshore
            data_out, #training data onshore
            model_def = modeldef, #model definition
            channels_off = [64,128,256], #channels for offshore(1DCNN)
            channels_on = [64,64], #channels for onshore(fully connected)
            epoch =  selected_epoch,#selected epoch
            batch_size = batch_size,
            control_points = control_points, #list of lat lon points to evaluate at these locations
            threshold = 0.1, #threshold above which to evaluate the model predictions
            verbose = False)

