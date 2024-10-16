# main.py
#This is where the experiment is run
import numpy as np
import experiment as exp

@exp.ex.automain
def run_experiment(MLDir,reg,reg_gaugeno,GaugeNo,windowthreshold,twindow,train_size,mask_size,test_size,batch_size,batch_size_on,
                   batch_size_deform,ts_dim,pts_dim,parts,z,h,channels_off,channels_on,channels_deform):
    # set seed and check cuda
    exp.set_seed_settings()

    # load the model
    AE = exp.BuildTsunamiAE()
    
    # Train
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

    # Training
    # t_array, red_d_array, red_dZ_array, dZ_array  = exp.read_memmap(what4 = 'train',
    #                                                     n_eve=n_eve,
    #                                                     nflood_grids=nflood_grids,
    #                                                     normalize=False,
    #                                                     standardize=False,
    #                                                     twindow = twindow,
    #                                                     windowthreshold = windowthreshold,
    #                                                     GaugeNo = GaugeNo, # list(range(35,44)), #for Catania
    #                                                     reg_gaugeno = reg_gaugeno #'38',
    #                                                     )
      
    # AE.pretrain(job = 'offshore',
    #            data = t_array,
    #            n = ts_dim,
    #            t = twindow, #pts_dim,
    #            z = z,
    #            h = h,
    #            channels = channels_off,
    #            nepochs = 600)
    
    # AE.pretrain(job = 'onshoreparts',
    #            data = red_d_array,
    #            parts = parts,
    #            n = nflood_grids,
    #            channels = channels_on,
    #            batch_size = batch_size_on,
    #            nepochs = 300)

    # AE.pretrain(job = 'deformfull',
    #         data = dZ_array,
    #         channels = channels_deform,
    #         z = z,
    #         batch_size = batch_size_deform,
    #         nepochs = 300)

    # AE.finetuneAE(data_in=t_array,
    #               data_deform=red_d_array,
    #               data_deformfull=dZ_array,
    #               data_out=red_d_array,
    #               n = nflood_grids,
    #               batch_size = batch_size_deform,
    #               nepochs = 200)

    # Testing
    event_list_path = f'{MLDir}/data/events/shuffled_events_test_{reg}_{test_size}.txt'
    event_list = np.loadtxt(event_list_path, dtype='str')
    n_eve = len(event_list)
    flood_mask = ~np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
    nflood_grids = np.count_nonzero(flood_mask)
    exp.ex.info["test_n_eve"] = n_eve 
        
    t_array, red_d_array, red_dZ_array, dZ_array = exp.read_memmap(what4 = 'test',
                                                         n_eve=n_eve,
                                                         nflood_grids=nflood_grids,
                                                         normalize=False,
                                                         standardize=False,)
    
    # epoch = '/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/scripts/Catania/sacred_logs/33/model_couple_off[64, 128, 256]_on[16, 128, 128]_minepoch_4005.pt' 
    #quick fix to load model path directly
    #MSE - 33
    #MCE - 27

    # AE.evaluateAE(data_in=t_array,
    #               data_deform=red_d_array,
    #               data_deformfull=dZ_array,
    #               data_out=red_d_array,
    #               batch_size = 1000,
    #               epoch=None,
    #               reg_gaugeno = reg_gaugeno,
    #                 )
    
    AE.evaluatePredic(
                  data_out=red_d_array,
                  reg_gaugeno = reg_gaugeno,
                    )
exp.run.stop() #stop neptune run and sync files

