# main.py
#This is where the experiment is run
import os
import numpy as np
import experiment as exp

@exp.ex.automain
def run_experiment(MLDir,reg,train_size,test_size,batch_size,batch_size_on,ts_dim,pts_dim,z,channels_off,channels_on):
    
    exp.set_seed_settings()

    # load training events related parameters
    event_list_path = f'{MLDir}/data/events/shuffled_events_{reg}_{train_size}.txt'
    event_list = np.loadtxt(event_list_path, dtype='str')
    n_eve = len(event_list)    
    
    flood_mask = ~np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{train_size}.npy')
    nflood_grids = np.count_nonzero(flood_mask)
    
    # log as info in sacred experiment, also used by read_memmap
    exp.ex.info["reg"] = reg
    exp.ex.info["event_list_path"] = event_list_path
    exp.ex.info["n_eve"] = n_eve
    exp.ex.info["nflood_grids"] = nflood_grids

    # load the model
    AE = exp.BuildTsunamiAE()

    # load training data
    t_array, red_d_array, red_dZ_array = exp.read_memmap(what4 = 'train', n_eve=n_eve, nflood_grids=nflood_grids)
   
    AE.pretrain(job = 'offshore',
               data = t_array,
               n = ts_dim,
               t = pts_dim,
               z = z,
               channels = channels_off)
    
    # AE.pretrain(job = 'onshore',
    #            data = red_d_array,
    #            n = nflood_grids,
    #            channels = channels_on,
    #            batch_size = batch_size_on)
    
    # AE.pretrain(job = 'deform',
    #            data = red_dZ_array,
    #            n = nflood_grids,
    #            channels = channels_on)

    # AE.finetuneAE(data_in=t_array,
    #               data_deform=red_dZ_array,
    #               data_out=red_d_array)

    # # test data 
    # event_list_path = f'{MLDir}/data/events/shuffled_events_test_{test_size}.txt'
    # event_list = np.loadtxt(event_list_path, dtype='str')
    # n_eve = len(event_list)
    # exp.ex.info["test_n_eve"] = n_eve 
        
    # t_array, red_d_array, red_dZ_array = exp.read_memmap(what4 = 'test', n_eve=n_eve, nflood_grids=nflood_grids)
    
    # AE.evaluateAE(data_in=t_array,
    #               data_deform=red_dZ_array,
    #               data_out=red_d_array,
    #               epoch=None,
    #               model_def=None)
      
exp.run.stop() #stop neptune run and sync files

