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
    ts_dim = len(GaugeNo) #gauges time series
elif reg == 'CT':
    GaugeNo = list(range(35,44)) #for Catania
    x_dim = 912
    y_dim = 2224

pts_dim = 480 #time steps

#load events data
event_list = np.loadtxt(f'{MLDir}/data/events/shuffled_events_{reg}_{size}.txt', dtype='str')
n_eve = len(event_list)
t_array = np.memmap(f'{MLDir}/data/processed/t_{reg}_{size}.dat',
                mode='r+',
                dtype=float,
                shape=(len(event_list), ts_dim, pts_dim))

#TODO:split as decided by ratio which is default here, can be sys arg or even event list files
data = t_array[:int(len(t_array)*0.65)]
test_data = t_array[int(len(t_array)*0.65):]

#directory to save model and outputs
if not os.path.exists(f'{MLDir}/model/{reg}/out/'):
    os.makedirs(f'{MLDir}/model/{reg}/out/')
if not os.path.exists(f'{MLDir}/model/{reg}/plot/'):
    os.makedirs(f'{MLDir}/model/{reg}/plot/')

#TODO: make these sys args as well, will be useful for hyperparameter tuning and loggin results aswell
batch_size = 50
num_epochs = 1001 #so that we can save the model at 1000 epochs
learning_rate = 0.0001
z = 64
channels = [64,128,256]

utils.trainAE(data,test_data,batch_size,num_epochs,learning_rate,n_eve,pts_dim,z,channels)