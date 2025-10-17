# main.py
#This is where the experiment is run
import numpy as np
import experiment as exp
from sklearn.linear_model import LinearRegression
import os

import numpy as np
from sklearn.linear_model import LinearRegression

def estimate_global_tweedie_p(Y, eps=1e-6):
    """
    Estimate Tweedie power p globally across all locations and events.

    Parameters
    ----------
    Y : ndarray of shape (n_events, n_locations)
        Observed inundation depths (non-negative, can include zeros)
    eps : float
        Small constant to avoid log(0)

    Returns
    -------
    p : float
        Estimated Tweedie power
    """
    # Flatten across events and locations
    Y_flat = Y.ravel()  # shape: (n_events * n_locations,)

    # Compute mean and variance per small block
    # For simplicity, we can compute variance across a block of consecutive samples
    # Here we treat the entire flattened array as one block
    mean_y = np.mean(Y_flat) + eps
    var_y = np.var(Y_flat, ddof=1) + eps

    # For a more stable estimate, you can divide into smaller blocks
    # Example: block_size = 1000
    block_size = 1000
    n_blocks = len(Y_flat) // block_size
    means = []
    vars_ = []
    for i in range(n_blocks):
        block = Y_flat[i*block_size:(i+1)*block_size]
        m = np.mean(block) + eps
        v = np.var(block, ddof=1) + eps
        means.append(m)
        vars_.append(v)
    means = np.array(means).reshape(-1,1)
    vars_ = np.array(vars_)

    # Fit linear regression: log(var) = c + p * log(mean)
    ols = LinearRegression(fit_intercept=True)
    ols.fit(np.log(means), np.log(vars_))

    p = ols.coef_[0]
    c = ols.intercept_
    print(f"Estimated global p = {p:.3f}, log(phi) = {c:.3f}")

    return p


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
    t_array, red_d_array, red_dZ_array, dZ_array  = exp.read_memmap(what4 = 'train',
                                                        n_eve=n_eve,
                                                        nflood_grids=nflood_grids,
                                                        normalize=False,
                                                        standardize=False,
                                                        twindow = twindow,
                                                        windowthreshold = windowthreshold,
                                                        GaugeNo = GaugeNo, # list(range(35,44)), #for Catania
                                                        reg_gaugeno = reg_gaugeno #'38',
                                                        )
      
    # AE.pretrain(job = 'offshore',
    #            data = t_array,
    #            n = ts_dim,
    #            t = twindow, #pts_dim,
    #            z = z,
    #            h = h,
    #            channels = channels_off,
    #            nepochs = 600)
    
    # AE.pretrain(job = 'deformfull',
    #         data = dZ_array,
    #         channels = channels_deform,
    #         z = z,
    #         batch_size = batch_size_deform,
    #         nepochs = 300)
    
    # AE.pretrain(job = 'onshoreparts',
    #            data = red_d_array,
    #            parts = parts,
    #            n = nflood_grids,
    #            channels = channels_on,
    #            batch_size = batch_size_on,
    #            nepochs = 300)

    # AE.finetuneAE(data_in=t_array,
    #               data_deform=red_d_array,
    #               data_deformfull=dZ_array,
    #               data_out=red_d_array,
    #               n = nflood_grids,
    #               batch_size = batch_size_deform,
    #               nepochs = 300)

    #calculate global tweedie p
    if task == 'tweedie_loss':
        p = estimate_global_tweedie_p(red_d_array)
        print(f'Using estimated global Tweedie p = {p:.3f} for training')

    AE.retuneED(
                job = 'withdeform', #nodeform or withdeform
                data_in=t_array,
                data_deformfull=dZ_array,
                data_out=red_d_array,
                n = nflood_grids,
                parts = parts,
                batch_size = batch_size_deform,
                nepochs = 1000,
)
    
#     AE.fulltuneEDerror(
#                 job = 'witherror', #nodeform or withdeform
#                 data_in=t_array,
#                 data_deformfull=dZ_array,
#                 data_out=red_d_array,
#                 n = nflood_grids,
#                 parts = parts,
#                 batch_size = batch_size_deform,
#                 nepochs = 1000,
# )
    
    # AE.fulltuneED(
    #             job = 'nodeform', #nodeform or withdeform
    #             data_in=t_array,
    #             data_deformfull=dZ_array,
    #             data_out=red_d_array,
    #             n = nflood_grids,
    #             parts = parts,
    #             batch_size = batch_size_deform,
    #             nepochs = 2000)
    
#     del t_array, red_d_array, red_dZ_array, dZ_array
#     del event_list, event_list_path, flood_mask, nflood_grids, n_eve
    
#     # Testing
#     event_list_path = f'{MLDir}/data/events/shuffled_events_test_{reg}_{test_size}.txt'
#     event_list = np.loadtxt(event_list_path, dtype='str')
#     n_eve = len(event_list)
#     flood_mask = ~np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
#     nflood_grids = np.count_nonzero(flood_mask)
#     exp.ex.info["test_n_eve"] = n_eve 
        
#     t_array, red_d_array, red_dZ_array, dZ_array = exp.read_memmap(what4 = 'test',
#                                                          n_eve=n_eve,
#                                                          nflood_grids=nflood_grids,
#                                                          normalize=False,
#                                                          standardize=False,)
    
#     epoch = '/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/model/CT/out/model_direct_off[64, 128, 256]_on[16, 128, 128]_minepoch_762.pt' 
#     #quick fix to load model path directly
#     #MSE - 33
#     #MCE - 27

#     AE.evaluateAE(data_in=t_array,
#                   data_deform=red_dZ_array,
#                   data_deformfull=dZ_array,
#                   data_out=red_d_array,
#                   batch_size = 1000,
#                   epoch=None,
#                   reg_gaugeno = reg_gaugeno,
#                     )

#     AE.evaluateED(data_in=t_array,
#                   data_deformfull=dZ_array,
#                   data_out=red_d_array,
#                   batch_size = 1000,
#                   epoch=None,
#                   reg_gaugeno = reg_gaugeno,
#                   )
    
exp.run.stop() #stop neptune run and sync files

