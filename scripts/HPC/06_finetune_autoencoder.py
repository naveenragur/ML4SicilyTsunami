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
split = 0.75
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

t_array = np.memmap(f'{MLDir}/data/processed/t_{reg}_{size}.dat',
                mode='r+',
                dtype=float,
                shape=(n_eve, ts_dim, pts_dim))

red_d_array = np.memmap(f'{MLDir}/data/processed/dflat_{reg}_{size}.dat',
                         mode='r+',
                         dtype=float,
                         shape=(n_eve, nflood_grids))

red_dZ_array = np.memmap(f'{MLDir}/data/processed/dZflat_{reg}_{size}.dat',
                            mode='r+',
                            dtype=float,
                            shape=(n_eve, nflood_grids))

#TODO:split as decided by ratio which is default here, can be sys arg or even event list files
# input data
data_in = t_array[:int(len(red_d_array)*split),:]
test_data_in = t_array[int(len(red_d_array)*split):,:]
data_deform =red_dZ_array[:int(len(red_d_array)*split),:]
test_data_deform = red_dZ_array[int(len(red_d_array)*split):,:]
data_out = red_d_array[:int(len(red_d_array)*split),:]
test_data_out = red_d_array[int(len(red_d_array)*split):,:]

#directory to save model and outputs
if not os.path.exists(f'{MLDir}/model/{reg}/out/'):
    os.makedirs(f'{MLDir}/model/{reg}/out/')
if not os.path.exists(f'{MLDir}/model/{reg}/plot/'):
    os.makedirs(f'{MLDir}/model/{reg}/plot/')

#TODO: make these sys args as well, will be useful for hyperparameter tuning and loggin results aswell
#learning parameters
batch_size = 20
num_epochs = 1000
learning_rate = 0.0005

#architecture parameters - offshore
# z = 64
# channels = [64,128,256]

#architecture parameters - onshore
# channels = [64,64]

#select a specific pretrained model #TODO: make this sys arg aswell
job = 'couple'
epoch_offshore = None
epoch_deform = None
epoch_onshore = None
interface_layers = 2 #no of layers in the interface between encoder and decoder
tune_nlayers = 1 #last n layer of encoder and first layer of decoder are also tunable

utils.finetuneAE(data_in, #training data offshore
            data_deform, #training data deformation
            data_out, #training data onshore
            test_data_in, #test data offshore
            test_data_deform, #test data deformation
            test_data_out, #test data onshore
            batch_size = batch_size,
            nepochs = num_epochs,
            lr = learning_rate,
            channels_off = [64,128,256], #channels for offshore(1DCNN)
            channels_on = [64,64], #channels for onshore(fully connected) and deformation
            couple_epochs = [epoch_offshore,epoch_deform,epoch_onshore], # epochs for offshore and onshore model coupling otherwise min loss epoch is used
            interface_layers=interface_layers, #no of inner interface layers,
            tune_nlayers = tune_nlayers, #no of layers to be tuned
            verbose = False)

