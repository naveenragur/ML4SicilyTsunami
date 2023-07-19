import os
import sys
# import glob

import numpy as np
import model_utils as utils

try:
    MLDir = os.getenv('MLDir')
    SimDir = os.getenv('SimDir')
    reg = sys.argv[1] #CT or SR
    size = sys.argv[2] #eventset size
    mode = sys.argv[3] #train or test
except:
    raise Exception("*** Must first set environment variable")

#set seed
np.random.seed(0)
split = 0.75
#TODO: could be read from a parameter file instead of hardcoding here
#dimensions and gauge numbers
if reg == 'SR':
    GaugeNo = list(range(53,58)) #rough pick for Siracusa
    x_dim = 1300  #lon
    y_dim = 948 #lat
    ts_dim = len(GaugeNo) #gauges time series
elif reg == 'CT':
    GaugeNo = list(range(35,44)) #for Catania
    x_dim = 912
    y_dim = 2224
    ts_dim = len(GaugeNo) #gauges time series
    
pts_dim = 480 #time steps

#load events data
event_list = np.loadtxt(f'{MLDir}/data/events/shuffled_events_{reg}_{size}.txt', dtype='str')
n_eve = len(event_list)
t_array = np.memmap(f'{MLDir}/data/processed/t_{reg}_{size}.dat',
                mode='r+',
                dtype=float,
                shape=(n_eve, ts_dim, pts_dim))

#TODO:split as decided by ratio which is default here, can be sys arg or even event list files
data = t_array[:int(len(t_array)*split)]
test_data = t_array[int(len(t_array)*split):]

#directory to save model and outputs
if not os.path.exists(f'{MLDir}/model/{reg}/out/'):
    os.makedirs(f'{MLDir}/model/{reg}/out/')
if not os.path.exists(f'{MLDir}/model/{reg}/plot/'):
    os.makedirs(f'{MLDir}/model/{reg}/plot/')

#TODO: make these sys args as well, will be useful for hyperparameter tuning and loggin results aswell
#learning parameters
batch_size = 50
num_epochs = 1001 #so that we can save the model at 1000 epochs
learning_rate = 0.0001

#architecture parameters - offshore
z = 64
channels_off = [64,128,256]

#architecture parameters - onshore
# channels_on = [64,64]

#train model of type onshore or offshore
job = 'offshore'
utils.pretrainAE(job, #offshore or onshore
            data, #training data 
            test_data, #test data 
            batch_size = batch_size,
            nepochs = num_epochs,
            lr = learning_rate,
            n = ts_dim, #no of offshore gauges
            t = pts_dim, #no of pts of time (480 time steps)
            z = z, #latent dim for offshore
            channels = channels_off, #channels for offshore(1DCNN) or #channels for onshore(fully connected)
            verbose = False)