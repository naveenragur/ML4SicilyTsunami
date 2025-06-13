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
    
    # Train Portion
    # load training events related parameters
    event_list_path = f'{MLDir}/data/events/shuffled_events_{reg}_{train_size}.txt'
    event_list = np.loadtxt(event_list_path, dtype='str')
    n_eve = len(event_list)    
    
    flood_mask = ~np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
    nflood_grids = np.count_nonzero(flood_mask)
    
    # log as info in sacred experiment, also used by read_memmap
    exp.ex.info["reg"] = reg
    exp.ex.info["event_list_path"] = event_list_path
    exp.ex.info["n_eve"] = n_eve
    exp.ex.info["nflood_grids"] = nflood_grids


    # Training Data
    t_array, red_d_array, dZ_array  = exp.read_memmap(what4 = 'train',
                                                        n_eve=n_eve,
                                                        nflood_grids=nflood_grids,
                                                        normalize=False,
                                                        standardize=False,
                                                        twindow = twindow,
                                                        windowthreshold = windowthreshold,
                                                        GaugeNo = GaugeNo, 
                                                        reg_gaugeno = reg_gaugeno,
                                                        )
      
    AE.fulltuneED(
                job = 'withdeform', 
                data_in=t_array,
                data_deformfull=dZ_array,
                data_out=red_d_array,
                n = nflood_grids,
                parts = parts,
                batch_size = batch_size_deform,
                nepochs = 1000,)
        
exp.run.stop() #stop neptune run and sync files
