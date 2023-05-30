import numpy as np
import pandas as pd

import xarray as xr
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary
from sklearn.model_selection import KFold
import torch.nn.functional as F
from sklearn.metrics import r2_score

import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline 

import folium
import folium.plugins
import pandas as pd
import xarray as xr
import branca
import branca.colormap as cm


#Read event list from file
event_list = np.loadtxt('../data/events/sample_events1212.txt', dtype='str')

#filter events with lower than threshold of 0.1 at atleast one station
offshore_threshold = 0.1
onshore_threshold = 0.25

allpts_max = np.loadtxt('../data/info/grid0_allpts87_alleve1212.offshore.txt', dtype='str',skiprows=1)
regions = ['CT','SR']

GaugeNo_CT = list(range(35,44)) #for Catania
GaugeNo_SR = list(range(53,58)) #for Siracusa
All_Gauges = GaugeNo_CT #+ GaugeNo_SR 

inun_info = np.loadtxt('../data/info/C_{:s}_alleve1212.onshore.txt'.format(regions[i]), dtype='str',skiprows=1)
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
np.savetxt('../data/events/shuffled_events_{:s}.txt'.format(regions[i]), event_list, fmt='%s')
np.savetxt('../data/events/train_events_{:s}.txt'.format(regions[i]), train_events, fmt='%s')
np.savetxt('../data/events/test_events_{:s}.txt'.format(regions[i]), test_events, fmt='%s')
