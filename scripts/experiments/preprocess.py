# main.py
#This is where the experiment is run
import os
import numpy as np
import experiment as exp

@exp.ex.automain
def run_experiment(MLDir,reg,train_size,reg_gaugeno,GaugeNo,offshore_threshold,onshore_threshold,split):
    
    exp.set_seed_settings()

    #sample events for building model
    train_size = exp.sample_events(wt_para = 'gridcount', #'LocationCount', 'mean_prob', 'importance', 'uniform_wt', 'gridcount'
                                    samples_per_bin = 15,
                                    bin_splits = 12)

    #offshore stats
    allpts_max = np.loadtxt(f'{MLDir}/data/info/grid0_allpts87_alleve53550.offshore.txt', dtype='str',skiprows=1)
    event_list = np.loadtxt(f'{MLDir}/data/events/sample_events{train_size}_{reg}_{reg_gaugeno}.txt', dtype='str')


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
      
exp.run.stop() #stop neptune run and sync files

