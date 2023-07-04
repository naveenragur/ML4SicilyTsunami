import os
import sys
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

#filter events with lower than threshold of 0.1 at atleast one station
offshore_threshold = 0.1
onshore_threshold = 0.25
split = 0.65

#TODO: change to read the master file with stats of all 53k events, here it uses the 1212 events
#TODO: selection of the number of gauge stations to use for inputs
#offshore
allpts_max = np.loadtxt(f'{MLDir}/data/info/grid0_allpts87_alleve53550.offshore.txt', dtype='str',skiprows=1)

if reg == 'CT':
    GaugeNo = list(range(35,44)) #for Catania
    MainGauge = str(41) #for Catania
elif reg == 'SR':
    GaugeNo = list(range(53,58)) #for Siracusa
    MainGauge = str(54) #for Siracusa

#TODO: Add results from sampling module say importance sampling
#Read event list from file
utils.sample_events(wt_para = 'gridcount', #'LocationCount', 'mean_prob', 'importance', 'uniform_wt', 'gridcount'
                  samples_per_bin = 15,
                  bin_splits = 12)

event_list = np.loadtxt(f'{MLDir}/data/events/sample_events{size}_{reg}_{MainGauge}.txt', dtype='str')
# event_list = np.loadtxt(f'{MLDir}/data/events/sample_events{size}.txt', dtype='str')

#select only the events as in the event_list
allpts_max = allpts_max[np.isin(allpts_max[:,0], event_list)]
Gauge_Max = allpts_max[:,GaugeNo]
maxPerEve = Gauge_Max.astype(float).max(axis=1)

#onshore
inun_info = np.loadtxt(f'{MLDir}/data/info/CDepth_{reg}_alleve53550.onshore.txt', dtype='str',skiprows=1)
inun_info = inun_info[np.isin(inun_info[:,0], event_list)]
Inun_Max = inun_info[:,2] #max inundation depth is the 3rd column


#filter events greater than thresholds for gauge and min inundation depth
offshore_check = maxPerEve>offshore_threshold
onshore_check = Inun_Max.astype(float)>onshore_threshold
print(len(event_list), len(offshore_check), len(onshore_check))

overall_check = offshore_check & onshore_check
event_list = event_list[overall_check]

#shuffle events
np.random.shuffle(event_list)

#split events in train and test and validation as 60:20:20
train_events = event_list[:int(len(event_list)*split)] 
test_events = event_list[int(len(event_list)*split):]

print(len(train_events), len(test_events))

#save events in file
np.savetxt(f'{MLDir}/data/events/shuffled_events_{reg}_{size}.txt', event_list, fmt='%s')
np.savetxt(f'{MLDir}/data/events/train_events_{reg}_{size}.txt', train_events, fmt='%s')
np.savetxt(f'{MLDir}/data/events/test_events_{reg}_{size}.txt', test_events, fmt='%s')
