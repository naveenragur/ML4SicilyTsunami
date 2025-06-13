# main.py
#This is where the experiment is run
import numpy as np
import experiment as exp
import os

@exp.ex.automain
def run_experiment(MLDir,reg,reg_gaugeno,GaugeNo,windowthreshold,twindow,train_size,mask_size,test_size,batch_size,batch_size_on,
                   batch_size_deform,ts_dim,pts_dim,parts,z,h,channels_off,channels_on,channels_deform,task,loss_type,asym_alpha,lr):
    # set seed and check cuda
    exp.set_seed_settings()

    # load the model
    AE = exp.BuildTsunamiAE()

    AE.check_dir()

    #create out directory
    if not os.path.exists(f'{MLDir}/model/{reg}/out'):
        os.makedirs(f'{MLDir}/model/{reg}/out')
    
    # Test Portion
    # load test events related parameters
    event_list_path = f'{MLDir}/data/events/shuffled_events_test_{reg}_{test_size}.txt'
    event_list = np.loadtxt(event_list_path, dtype='str')
    n_eve = len(event_list)
    flood_mask = ~np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
    nflood_grids = np.count_nonzero(flood_mask)
    exp.ex.info["test_n_eve"] = n_eve 

    # Test Data    
    t_array, red_d_array, dZ_array = exp.read_memmap(what4 = 'test',
                                                         n_eve=n_eve,
                                                         nflood_grids=nflood_grids,
                                                         normalize=False,
                                                         standardize=False,)
        
    AE.evaluateEDerror( 
                   job = 'mc_dropout',
                   data_in=t_array,
                   data_deformfull=dZ_array,
                   data_out=red_d_array,
                   batch_size = 100,
                   epoch=None,
                   reg_gaugeno = reg_gaugeno,
                     )
        
exp.run.stop() #stop neptune run and sync files
