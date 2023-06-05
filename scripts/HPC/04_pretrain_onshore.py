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
except:
    raise Exception("*** Must first set environment variable")

#set seed
np.random.seed(0)

#TODO: could be read from a parameter file instead of hardcoding here
#dimensions and gauge numbers
if reg == 'SR':
    GaugeNo = list(range(53,58)) #rough pick for Siracusa
    x_dim = 1300  #lon
    y_dim = 948 #lat
    ts_dim = len(GaugeNo) #no of gauges time series
elif reg == 'CT':
    GaugeNo = list(range(35,44)) #for Catania
    x_dim = 912
    y_dim = 2224
    ts_dim = len(GaugeNo) 
    
pts_dim = 480 #time steps

#load events data
event_list = np.loadtxt(f'{MLDir}/data/events/shuffled_events_{reg}_{size}.txt', dtype='str')
flood_mask = ~np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{size}.npy')
nflood_grids = np.count_nonzero(flood_mask)
print(f'Number of flooded grids: {nflood_grids}')

n_eve = len(event_list)
print(f'Number of events: {n_eve}')

red_d_array = np.memmap(f'{MLDir}/data/processed/dflat_{reg}_{size}.dat',
                         mode='r+',
                         dtype=float,
                         shape=(n_eve, nflood_grids))

#TODO:split as decided by ratio which is default here, can be sys arg or even event list files
# input data
data = red_d_array[:int(len(red_d_array)*0.65),:]
test_data = red_d_array[int(len(red_d_array)*0.65):,:]

#directory to save model and outputs
if not os.path.exists(f'{MLDir}/model/{reg}/out/'):
    os.makedirs(f'{MLDir}/model/{reg}/out/')
if not os.path.exists(f'{MLDir}/model/{reg}/plot/'):
    os.makedirs(f'{MLDir}/model/{reg}/plot/')

#TODO: make these sys args as well, will be useful for hyperparameter tuning and loggin results aswell
#learning parameters
batch_size = 20
num_epochs = 3001
learning_rate = 0.00005

#architecture parameters - offshore
# z = 64
# channels_off = [64,128,256]

#architecture parameters - onshore
channels_on = [64,64]

#train model of type onshore or offshore
job = 'onshore'
utils.pretrainAE(job, #offshore  or onshore or couple
            data, #training data 
            test_data, #test data 
            batch_size = batch_size,
            nepochs = num_epochs,
            lr = learning_rate,
            n = nflood_grids, #no of offshore gauges or inundated grids
            t = None, #no of pts of time (480 time steps)
            z = None, #latent dim for offshore only
            channels = channels_on, #channels for offshore(1DCNN) or #channels for onshore(fully connected)
            verbose = False)

