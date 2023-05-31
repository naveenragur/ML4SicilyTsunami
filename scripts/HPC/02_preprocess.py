import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import KFold
import torch.nn.functional as F
from sklearn.metrics import r2_score

try:
    MLDir = os.getenv('MLDir')
    SimDir = os.getenv('SimDir')
    reg = sys.argv[1] #CT or SR
    size = sys.argv[2] #eventset size
except:
    raise Exception("*** Must first set environment variable")

#Read event list from file
event_list = np.loadtxt(f'{MLDir}/data/events/sample_events{size}.txt', dtype='str')



#filter events with lower than threshold of 0.1 at atleast one station
offshore_threshold = 0.1
onshore_threshold = 0.25

allpts_max = np.loadtxt(f'{MLDir}/data/info/grid0_allpts87_alleve1212.offshore.txt', dtype='str',skiprows=1)
#reg = ['CT','SR']

GaugeNo_CT = list(range(35,44)) #for Catania
GaugeNo_SR = list(range(53,58)) #for Siracusa
All_Gauges = GaugeNo_CT #+ GaugeNo_SR 

inun_info = np.loadtxt(f'{MLDir}/data/info/C_{reg}_alleve1212.onshore.txt', dtype='str',skiprows=1)
Inun_Max = inun_info[:,2]

Gauge_Max = allpts_max[:,All_Gauges]
maxPerEve = Gauge_Max.astype(float).max(axis=1)

#filter events greater than thresholds for gauge and min inundation depth
offshore_check = maxPerEve>offshore_threshold
onshore_check = Inun_Max.astype(float)>onshore_threshold
overall_check = offshore_check & onshore_check
event_list = event_list[overall_check]

#set seed
np.random.seed(123)

#shuffle events
np.random.shuffle(event_list)

#split events in train and test and validation as 60:20:20
train_events = event_list[:int(len(event_list)*0.65)] 
test_events = event_list[int(len(event_list)*0.65):]

print(len(train_events), len(test_events))

#save events in file
np.savetxt(f'{MLDir}/data/events/shuffled_events_{reg}.txt', event_list, fmt='%s')
np.savetxt(f'{MLDir}/data/events/train_events_{reg}.txt', train_events, fmt='%s')
np.savetxt(f'{MLDir}/data/events/test_events_{reg}.txt', test_events, fmt='%s')
