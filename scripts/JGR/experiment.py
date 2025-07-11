#experiment.py
#This is where the model is built with config parameters and the ML train and evaluation ie the experiment is designed
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import gaussian_kde

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import contextily as cx

import neptune
from neptune.integrations.sacred import NeptuneObserver
from sacred import Experiment
from sacred.observers import FileStorageObserver

# Set up Sacred experiment
ex = Experiment("autoencoder_experiment")

# Set up Sacred observers and the neptune instance and logger. Turn off if using just some function from this file
ex.observers.append(FileStorageObserver.create("sacred_logs"))

run = neptune.init_run(project="naveenragur/ML4Sicily",
                    source_files=["experiment.py","main.py","parameters.json","run.sbatchCT","train.py","test.py"],
                    api_token=os.getenv('Neptune_api_token'),
                    )
ex.observers.append(NeptuneObserver(run=run))


# Set up the configuration parameters that can be easily called and modified when running the experiment, configuration in run script superceeds these experiment config parameters
@ex.config
def config():
    # Seed for reproducibility
    seed = 0

    # Define the data path
    MLDir = "/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/"
    SimDir = "/mnt/beegfs/nragu/tsunami/ML4SicilyTsunami/data/simu/"

    # #threshold for preprocess section in preprocess
    # offshore_threshold = 0.1
    # onshore_threshold = 0.25

    # Define the region and data size parameters
    reg = "CT" #CT or SR
    train_size = "9999" #eventset size for training & building the model
    mask_size = "999" #eventset size for masking
    test_size = "0" #eventset size for testing 
    off_size = '9999' #offshore size used for pretraining to pick coupling model encoders
    deform_size = '9999' #deformation size used for pretraining to pick coupling model encoders

    # Define the model region related size/architecture
    if reg == 'SR':
        GaugeNo = list(range(53,58)) #rough pick for Siracusa
        x_dim = 1300  #lon
        y_dim = 948 #lat
        ts_dim = len(GaugeNo) #no of gauges time series
        reg_gaugeno = str(54)
        control_points = [[37.01,15.29],
            [37.06757,15.28709],
            [37.05266,15.26536],
            [37.03211,15.28632]]
    elif reg == 'CT':
        GaugeNo = list(range(35,44)) #for Catania
        x_dim = 912
        y_dim = 2224
        ts_dim = len(GaugeNo)
        reg_gaugeno = str(38)
        control_points =  [[37.5022,15.0960],
            [37.48876,15.08936],
            [37.47193,15.07816],
            [37.46273,15.08527],
            [37.46252,15.08587],
            [37.45312,15.07874],
            [37.42821,15.08506],
            [37.40958,15.08075],
            [37.38595,15.08539],
            [37.35084,15.08575],
            [37.33049,15.07029],
            [37.40675,15.05037]
        ]
    pts_dim = 480 #time series length

    # Define hyperparameters and training configurations
    lr = 0.0035

    es_gap = 200
    step_size = 300
    gamma = 0.9
    
    batch_size = 300
    nepochs = 3500
    folds = 4 #sets the split between train and test 0.75:0.25

    #loss type
    loss_type = None
    asym_alpha = None
    asym_beta = None

    # Additional variables for the model architecture
    z = 64 #latent space for offshore and deform
    h = 10 #hidden state for LSTM if used
    parts = 64 #no of splits or parts for onshore
    channels_off = [64,128,256]
    channels_on = [16,128,128]
    channels_deform = [16,32,64,128]

    # Define the epoch number for loading the pretrained model
    epoch_offshore = None
    epoch_deform = None
    epoch_onshore = None

    # Define the number of layers to be tuned
    interface_layers = 2 #no of layers in the interface between encoder and decoder
    tune_nlayers = 2 #last n layer of encoder and first layer of decoder are also tunable

    #for reading data
    windowthreshold = 0.1
    twindow = 480 #in mins

    #for evaluation
    threshold = 0.2 #threshold for flooded or not in evaluation

    #to define folder path
    task = None

# To setup the region for the experiment
@ex.named_config
def SR():
    reg = "SR"

@ex.named_config
def CT():
    reg = "CT"

# Set seed and torch set
@ex.capture
def set_seed_settings(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')
    print('cuda is:',torch.cuda.is_available())

# Set up the data reading function
@ex.capture
def read_memmap(MLDir,
                reg,
                task,
                train_size,
                test_size,
                ts_dim,
                pts_dim,
                x_dim,
                y_dim,
                n_eve,
                nflood_grids,
                what4 = 'train',
                normalize = False,
                standardize = False,
                twindow = None, #for reduced data period
                windowthreshold = 0.1,
                GaugeNo = list(range(35,44)), #for Catania
                reg_gaugeno = '54',) :
    
    if what4 == 'train':
        size = str(train_size)
    elif what4 == 'test':
        size = str(test_size)
    
    print('reading memmap for', what4, 'data','size:', size)

    #read the data
    t_array = np.memmap(f'{MLDir}/data/processed/t_{reg}_{size}.dat',
            mode='r',
            dtype=float,
            shape=(n_eve, ts_dim, pts_dim))
    
    if twindow is not None:
        # Find start and end index of twindow based on threshold of 0.1 for the specified gauge
        rep_gidx = GaugeNo.index(int(reg_gaugeno))
        print("Using", twindow, "window for gauge:", reg_gaugeno, "index:", rep_gidx)

        # Get start and end index (start + twindow or pts_dim) of twindow based on threshold test for ts of rep_gidx gauge
        # start_idx = np.argmax(np.abs(t_array[:, rep_gidx, :]) > windowthreshold, axis=1)
        start_idx = []
        for e in range(n_eve):
            start_idx.append(np.argmax(np.any(np.abs(t_array[e,:,:]) > .1, axis=0)))
        #add twindow to start index
        start_idx = np.array(start_idx)
        end_idx = start_idx + twindow - 1

        # For end_idx > pts_dim, set to pts_dim - 1
        end_idx[end_idx > pts_dim - 1] = pts_dim - 1
        len_idx = end_idx - start_idx + 1

        # Set a new window for all gauges and maintain the length of twindow
        t_array_temp = np.zeros((n_eve, ts_dim, twindow))
        for i in range(n_eve):
            t_array_temp[i, :, 0:len_idx[i]] = t_array[i, :, start_idx[i]:end_idx[i] + 1]  # Fix indexing

        # # Plot a sample of ts for all gauges for the first event
        # plt.figure(figsize=(15, 5))
        # plt.plot(t_array_temp[0].T)  # Plot t_array_temp instead of t_array
        # plt.title(f"Time series for all gauges for the first event")
        # plt.savefig(f'{MLDir}/model/{reg}/{task}/plot/ts_{reg}_{size}_firstevent.png')
        # plt.clf()
        
        #save start and end index to file
        np.savetxt(f'{MLDir}/data/processed/twindow_{reg}_{size}.txt', np.column_stack((start_idx, end_idx)), delimiter=',')

    red_d_array = np.memmap(f'{MLDir}/data/processed/dflat_{reg}_{size}.dat',
                            mode='r',
                            dtype=float,
                            shape=(n_eve, nflood_grids))

    red_dZ_array = np.memmap(f'{MLDir}/data/processed/dZflat_{reg}_{size}.dat',
                            mode='r',
                            dtype=float,
                            shape=(n_eve, nflood_grids))
    
    dZ_array = np.memmap(f'{MLDir}/data/processed/dZ_{reg}_{size}.dat',
                            mode='r',
                            dtype=float,
                            shape=(n_eve,y_dim,x_dim))
    
    tmin,tmax = -5,5
    dmin,dmax = 0,22
    dZmin,dZmax = -5,5

    if normalize:
        print('normalizing data')
        # # Perform normalization and standardization in a single step
        # t_array -= t_array.min()
        # t_array /= t_array.max()

        # red_d_array -= red_d_array.min()
        # red_d_array /= red_d_array.max()

        # red_dZ_array -= red_dZ_array.min()
        # red_dZ_array /= red_dZ_array.max()
        # Apply normalization using predefined values
        t_array -= tmin
        t_array /= (tmax - tmin)

        red_d_array -= dmin
        red_d_array /= (dmax - dmin)

        red_dZ_array -= dZmin
        red_dZ_array /= (dZmax - dZmin)

    if standardize:
        print('standardizing data')
        if what4 == 'train':
            # Calculate mean and standard deviation for training data
            tmn, tsd = np.mean(t_array), np.std(t_array)
            dmn, dsd = np.mean(red_d_array), np.std(red_d_array)
            dZmn, dZsd = np.mean(red_dZ_array), np.std(red_dZ_array)

            std_para = [tmn, tsd, dmn, dsd, dZmn, dZsd]

            # Save to file
            np.savetxt(f'{MLDir}/data/processed/std_para_{reg}_{size}.txt', std_para, delimiter=',')
        else:
            # Load mean and standard deviation for test data
            std_para = np.loadtxt(f'{MLDir}/data/processed/std_para_{reg}_{train_size}.txt', delimiter=',')
            tmn, tsd, dmn, dsd, dZmn, dZsd = std_para

        # Apply standardization using calculated mean and standard deviation
        t_array -= tmn
        t_array /= tsd

        red_d_array -= dmn
        red_d_array /= dsd

        red_dZ_array -= dZmn
        red_dZ_array /= dZsd

    if twindow is not None:
        return t_array_temp, red_d_array, dZ_array
    else:
        return t_array, red_d_array, dZ_array

# Set up the function to get the index from lat/lon for inundation locations
@ex.capture
def get_idx_from_latlon(locations,reg,MLDir,SimDir,mask_size):  
    #get first event to get lat lon
    firstevent = np.loadtxt(f'{MLDir}/data/events/sample_events53550.txt',dtype='str')[0]
    D_grids = xr.open_dataset(f'{SimDir}/{firstevent}/{reg}_flowdepth.nc')
    zero_mask = np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{mask_size}.npy')
    non_zero_list = np.argwhere(~zero_mask).tolist()

    #iterate over list locations
    indices = []
    for loc in locations: 
        #get index of lat lon
        lat_idx = np.argmin(np.abs(D_grids.lat.values - loc[0]))
        lon_idx = np.argmin(np.abs(D_grids.lon.values - loc[1]))

        #get idx in non zero mask list from lat_idx and lon_idx
        idx = non_zero_list.index([lat_idx,lon_idx])
        indices.append(idx)

    # return lat_idx, lon_idx, idx
    # print(indices)
    return indices

#Set up the function to calculate the scores and metrics
def calc_scores(true,pred,locindices,threshold=0.2):
    truecopy = true.copy()
    #only test where there is significant flooding
    truecopy[truecopy<threshold] = 0
    pred[pred<threshold] = 0
    mse_val = mean_squared_error(truecopy,pred)
    r2_val = r2_score(truecopy,pred)
    Gfit_val = Gfit_one(truecopy,pred)
    l2n_val = l2norm(truecopy,threshold)
    pt_er = truecopy[locindices] - pred[locindices]

    #calculate Aidan's no K and k small
    # ratio = true[true>=threshold]/pred[true>=threshold]    
    # logK = np.exp((np.log(ratio)).mean()) # Aidan's no K
    # logksmall = (np.mean((np.log(ratio))**2) - (logK**2))**0.5
    # Kcap = np.exp(logK)
    # Ksmall = np.exp(logksmall)
    return mse_val,r2_val,truecopy[locindices],pred[locindices],pt_er,Gfit_val,l2n_val#,Kcap,Ksmall

def calc_errors(residual,locindices,var=True):
    if var:
        var = residual.copy()
        pt_var = var[locindices]
        pt_var[pt_var<0] = 0
        pt_sigma = pt_var**0.5
    else:
        sigma = residual.copy()
        pt_sigma = sigma[locindices]
        pt_sigma[pt_sigma<0] = 0
        
    return pt_sigma

def Gfit(obs, pred): #a normalized least-squares per event in first dimensions
    # print('obs shape', obs.shape,obs.shape[0])
    Gtable = np.zeros(obs.shape[0])
    for i in range(obs.shape[0]):      
        obs_i = np.array(obs[i])
        pred_i = np.array(pred[i])
        Gvaluei = 1 - (2*np.sum(obs_i*pred_i)/(np.sum(obs_i**2)+np.sum(pred_i**2)))
        Gtable[i] = Gvaluei
    return Gtable

def Gfit_one(obs, pred): #a normalized least-squares
    obs = np.array(obs)
    pred = np.array(pred)
    Gvalue = 1 - (2*np.sum(obs*pred)/(np.sum(obs**2)+np.sum(pred**2)))
    return Gvalue

def l2norm(obs,thresh):
    obs = np.array(obs)
    return np.sqrt(np.sum(obs**2)/len(obs>thresh))    
    
#Main Class used to design ML architecture for the tsunami emulator
class EncoderDecoder(nn.Module):  #experiment configuration superceeds the below defaults
    def __init__(self, 
                x, #input wave data
                y, #output depth data
                xy, #output size
                ninputs=5, #number of input channels or gauges
                ch_list = [32,64,96], #number of channels in each layer,
                zlist=[16, 256, 256], #number of channels in each layer for onshore
                df_list=[16, 32, 64, 128], #number of channels in each layer for deformation
                zdim = 50, #latent space for offshore and deform
                parts = 64, #number of splits or parts for onshore
                ):
        super(EncoderDecoder, self).__init__()

        self.ch_list = ch_list
        self.parts = parts
        self.x = x
        self.y = y
        self.xy = xy
        self.split_size = (xy // parts)+1
        self.zdim = zdim

        # define offshore encoder layers
        self.offshore_encoder = nn.Sequential(
            nn.Conv1d(ninputs, ch_list[0], kernel_size=3, padding=1),   
            nn.LeakyReLU(negative_slope=0.5,inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(ch_list[0], ch_list[1], kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.5,inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(ch_list[1], ch_list[2], kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.5,inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
        ) 

        self.offshore_fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(zdim),
        )

        # define deform encoder layers
        self.deform_encoder = nn.Sequential(
            nn.Conv2d(1, df_list[0], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.5, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(df_list[0], df_list[1], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.5, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(df_list[1], df_list[2], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.5, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(df_list[2], df_list[3], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.5, inplace=True),
            nn.Dropout(0.1),
        )

        self.deform_fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(zdim),
        )

        self.dropout = nn.Dropout(0.5)

        self.connect = nn.Sequential(
                                nn.LazyLinear(128),
                                nn.LeakyReLU(inplace=True),
                                nn.LazyLinear(128),
                                nn.LeakyReLU(inplace=True),
            )

        self.onshore_global_decoder = nn.Sequential(
            nn.LazyLinear(zlist[-2]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.LazyLinear(zlist[-1]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.LazyLinear(zlist[0]*parts),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.onshore_split_decoders = nn.ModuleList()
        for i in range(parts):
            decoder = nn.Sequential(
                nn.LazyLinear(self.split_size),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            )
            self.onshore_split_decoders.append(decoder)
       
    def forward(self, x, dz):
        #encode offshore time series to latent space
        x = self.offshore_encoder(x)
        x = self.offshore_fc(x)
        #encode deformation to latent space
        dz = dz.unsqueeze(1)  
        dz = self.deform_encoder(dz)
        dz = self.deform_fc(dz)
        dz = dz.squeeze(1) #to reduce one of the dimension
        #common latent space from from encoders
        z = torch.cat((x, dz), dim=1)
        z = self.dropout(z)
        z = self.connect(z)
        #decode to onshore global intermediate latent space
        global_decoded = self.onshore_global_decoder(z)
        split_global_decoded = torch.chunk(global_decoded, self.parts, dim=1) #z[0]*parts 
        # Decode each split independently
        decoded_splits = []
        for i in range(self.parts): #z[0]*parts to 600k/parts
            decoded_split = self.onshore_split_decoders[i](split_global_decoded[i])
            decoded_splits.append(decoded_split)
        # Concatenate the decoded parts
        y = torch.cat(decoded_splits, dim=1) #600k/parts to 600k
        y = y[:, :self.xy] #crop to original size
        return y

#Main Class used to build the model, train and evaluate it
class BuildTsunamiAE():
    @ex.capture
    def __init__(self,
                MLDir =None, #directory to save model
                reg =None, #regularization
                train_size =None, #training size
                test_size =None, #test size
                task =None, #task type
                asym_alpha = None, #asymmetry parameter
                asym_beta = None, #asymmetry parameter
                step_size = 100, #step size for scheduler
                gamma = 0.1, #gamma for schedule""r
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                # criteria = nn.L1Loss(reduction='mean'), #loss function
                criteria = nn.MSELoss(reduction='mean'), #loss function
                # criteria = nn.HuberLoss(reduction='mean', delta=1.0), #loss function
                verbose = False,
                es_gap = 1000,
                seed = 0, 
                ):
        self.model = None
        self.optimizer = None
        self.step_size = step_size
        self.gamma = gamma
        self.device = device
        self.MLDir = MLDir
        self.reg = reg
        self.train_size = train_size
        self.test_size = test_size
        self.verbose = verbose
        self.es_gap = es_gap
        self.seed = seed
        self.criterion = criteria
        self.task = task
        self.asym_alpha = asym_alpha
        self.asym_beta = asym_beta

    #check if directory exists and create if not
    def check_dir(self):
        if not os.path.exists(f'{self.MLDir}/model/{self.reg}/{self.task}'):
            os.makedirs(f'{self.MLDir}/model/{self.reg}/{self.task}')
        if not os.path.exists(f'{self.MLDir}/model/{self.reg}/{self.task}/plot'):
            os.makedirs(f'{self.MLDir}/model/{self.reg}/{self.task}/plot')
        if not os.path.exists(f'{self.MLDir}/model/{self.reg}/{self.task}/out'):
            os.makedirs(f'{self.MLDir}/model/{self.reg}/{self.task}/out')
    
    #configure the optimizer to manage parameters updates during training
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    #configure the scheduler to manage the learning rate
    def configure_scheduler(self):
            self.scheduler = StepLR(self.optimizer, self.step_size, self.gamma)

    #loads dataset as a tensor dataset into device
    def dataloader(self,dataset):
        tensor_dataset = TensorDataset(torch.Tensor(dataset)) #alternatively use TensorDataset(torch.from_numpy(dataset))
        return DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle = False)
    
    def get_fold_data(self, any_data, k=0):
        """
        Split data into train and test for k-th fold out of 'folds' total.
        Each fold rotates the test set, with the rest used for training.

        Args:
            any_data: Full dataset (e.g., list, array, or tensor)
            folds (int): Number of folds (e.g., 4 means 25% test split)
            k (int): Fold index (0 <= k < folds)

        Returns:
            train_data, test_data
        """
        
        assert 0 <= k < self.folds, "k must be in range [0, folds-1]"

        n_total = len(any_data)
        fold_size = n_total // self.folds

        test_start = k * fold_size
        test_end = (k + 1) * fold_size if k != self.folds - 1 else n_total

        # Create indices
        test_indices = list(range(test_start, test_end))
        train_indices = list(range(0, test_start)) + list(range(test_end, n_total))

        #print range for fold
        print(f"Fold {k}: Train indices {train_indices[0]} to {test_start} and {test_end+1} to {n_total}")
        print(f"Fold {k}: Test indices {test_start} to {test_end}")

        # Slice data
        train_data = any_data[train_indices]
        test_data = any_data[test_indices]

        return train_data, test_data

    #get dataloader for train and test based on folds for using complete dataset for training
    def get_dataloader(self, any_data, k):
        """
        Depending on the value of k, the data portion is mixed in chunks and used for training and testing
        """

        train_data, test_data = self.get_fold_data(any_data,k)
        train_loader = self.dataloader(train_data[0:int(len(train_data)*0.99)]) #train data 
        test_loader = self.dataloader(test_data) #test data not used in training but helps overfit check and early stopping procedures

        return train_loader, test_loader

    #save and plot loss during training   
    def plot_save_loss(self,k):
            plt.plot(self.train_epoch_losses, color='blue')
            plt.plot(self.test_epoch_losses, color='green')
            plt.axvline(x=self.min_epoch, color='black', linestyle='--')
            plt.text(self.min_epoch, .001, self.min_epoch, fontsize=12)
            plt.legend(['train', 'test'], loc='upper left')
            plt.title(f"Training loss for fold:{k}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.yscale('log')
            if self.job == 'withdeform':
                plt.savefig(f'{self.MLDir}/model/{self.reg}/{self.task}/plot/{self.job}_loss_off{self.channels_off}_on{self.channels_on}_{self.train_size}_fold_{k}.png')
                ex.add_artifact(f'{self.MLDir}/model/{self.reg}/{self.task}/plot/{self.job}_loss_off{self.channels_off}_on{self.channels_on}_{self.train_size}_fold_{k}.png')
            else:
                plt.savefig(f'{self.MLDir}/model/{self.reg}/{self.task}/plot/{self.job}_loss_ch_{self.channels}_{self.train_size}_fold_{k}.png')   
                ex.add_artifact(f'{self.MLDir}/model/{self.reg}/{self.task}/plot/{self.job}_loss_ch_{self.channels}_{self.train_size}_fold_{k}.png')
            plt.clf()

    #custom loss function to handle underpredictions
    def asym_mse_scale(self, recon, true):
        """
        Loss function where mse is over inundated locations only.

        Parameters:
        recon (tensor): Reconstructed predictions - batch x locations.
        true (tensor): Numerical simulations - batch x locations.

        Returns:
        tensor: Computed loss.
        """
        #set alpha based on depth of inundation as (1 + depth) to handle zero depth
        gamma = 1 + true ** 1
        loss = torch.mean(torch.where(recon < true, gamma * (true - recon) ** 2,  (recon - true) ** 2))
        return loss 

    #To sample the model with dropout layers
    @ex.capture    
    def mc_dropout_inference(
        self,
        wave_input,
        deform_input,
        control_points=None,
        num_samples=50,
        ):
        """
        Perform Monte Carlo Dropout inference using models across folds.

        Returns:
            mean_prediction: Mean prediction [batch, locations]
            sigma_prediction: Standard deviation [batch, locations]
            location_samples: Optional subset at specified locations [samples, batch, selected_locs]
        """
        all_predictions = []

        if control_points is not None:
            locindices = get_idx_from_latlon(control_points)
        else:
            locindices = None

        for k in range(self.folds):
            model_path = f'{self.MLDir}/model/{self.reg}/{self.task}/out/model_withdeform_off{self.channels_off}_on{self.channels_on}_minepoch_{self.train_size}_fold_{k}.pt'
            fold_model = torch.load(model_path, map_location=self.device)
            fold_model.train()

            fold_samples = []

            with torch.no_grad():
                for _ in range(num_samples):
                    output = fold_model(wave_input, deform_input)  # [batch, locations]
                    fold_samples.append(output.detach().cpu())  # Move to CPU

            fold_predictions = torch.stack(fold_samples)  # [num_samples, batch, locations]
            all_predictions.append(fold_predictions)

            del fold_model
            torch.cuda.empty_cache()

        predictions = torch.cat(all_predictions, dim=0)  # [folds * samples, batch, locations]
        mean_prediction = predictions.mean(dim=0)  # [batch, locations]
        sigma_prediction = predictions.std(dim=0)

        if locindices is not None:
            location_samples = predictions[:, :, locindices]
        else:
            location_samples = predictions

        return mean_prediction.to(self.device), sigma_prediction.to(self.device), location_samples


    #tune the encoder-decoder model parameters
    @ex.capture 
    def fulltuneED(self,
            job,
            data_in, #training data offshore
            data_deformfull, #training data deformation
            data_out, #training data offshore
            folds, #number of folds, also sets the split between train and test
            batch_size, #batch size onshore
            nepochs,
            lr,
            channels_off = [64,128,256], #channels for offshore(1DCNN)
            channels_on = [64,64], #channels for onshore(fully connected)
            channels_deform = [16,32,64,128], #channels for deformation(1DCNN)
            loss_type = None,
            z= None, #latent dim 
            ts_dim = None, #time series dim
            parts = None,
            x_dim = None,
            y_dim = None,
            n = None,
            ):
        super().__init__()

        #data
        self.data_in = data_in
        self.data_deformfull = data_deformfull
        self.data_out = data_out

        #model structure
        self.ninputs = ts_dim
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.parts = parts
        self.xy = n
        
        #hyperparameters
        self.folds = folds
        self.batch_size = batch_size
        self.nepochs = nepochs
        self.lr = lr

        #model parameters/architecture
        self.channels_off = channels_off
        self.channels_on = channels_on
        self.channels_deform = channels_deform
        self.z = z
        self.job = job #with deform or no deform

        #train model for multiple folds
        for k in range(self.folds):

            # Initialize model,criteria	and optimizer
            if self.job == 'withdeform':
                self.model = EncoderDecoder(ninputs=self.ninputs,
                                            x=self.x_dim,
                                            y=self.y_dim,
                                            xy=self.xy,
                                            ch_list=self.channels_off,
                                            zlist=self.channels_on,
                                            df_list=self.channels_deform,
                                            zdim=self.z,
                                            parts=self.parts)
            
            if loss_type == 'mse_asym_scale':
                self.criterion = self.asym_mse_scale
            elif loss_type == None:
                print('using default loss function MSELoss')     

            print(self.criterion)

            self.model.to(self.device)
            self.configure_optimizers()
            self.configure_scheduler()

            #load data for folds
            train_loader_in, test_loader_in = self.get_dataloader(self.data_in,k) 
            train_loader_deformfull, test_loader_deformfull = self.get_dataloader(self.data_deformfull,k) 
            train_loader_out,test_loader_out = self.get_dataloader(self.data_out,k) 
            self.train_epoch_losses, self.test_epoch_losses = [], []

            # Train model
            for epoch in range(self.nepochs):
                if epoch == 2:
                    #calculate no of param and print
                    print('no of total param:',sum(p.numel() for p in self.model.parameters() if p.requires_grad))
                train_loss,test_loss = 0, 0
                for batch_idx,(batch_data_in,batch_data_deformfull,batch_data_out) in enumerate(zip(train_loader_in,train_loader_deformfull,train_loader_out)):
                    self.optimizer.zero_grad()
                    batch_data_in = batch_data_in[0].to(self.device)
                    batch_data_deformfull = batch_data_deformfull[0].to(self.device)
                    batch_data_out = batch_data_out[0].to(self.device)
                    recon_data = self.model(batch_data_in,batch_data_deformfull)
                    loss = self.criterion(recon_data, batch_data_out)
                    train_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                        
                for batch_idx,(batch_data_in,batch_data_deformfull,batch_data_out)  in enumerate(zip(test_loader_in,test_loader_deformfull,test_loader_out)):
                    batch_data_in = batch_data_in[0].to(self.device)
                    batch_data_deformfull = batch_data_deformfull[0].to(self.device)
                    batch_data_out = batch_data_out[0].to(self.device)
                    recon_data = self.model(batch_data_in,batch_data_deformfull)
                    tloss = self.criterion(recon_data, batch_data_out)
                    test_loss += tloss.item()

                avg_train_ls = train_loss/len(train_loader_in)
                avg_test_ls = test_loss/len(test_loader_in)

                #log to neptune
                log2neptune = True
                if log2neptune:
                    run[f'train/{self.job}/epochloss'].append(avg_train_ls)
                    run[f'test/{self.job}/epochloss'].append(avg_test_ls)

                if self.verbose:
                    print(f'fold:{k},epoch:{epoch},training loss:{avg_train_ls:.5f},test loss:{avg_test_ls:.5f}', end='\r')
                
                self.train_epoch_losses.append(avg_train_ls)
                self.test_epoch_losses.append(avg_test_ls)
                
                #save model a sepcific intermediate epoch
                # if epoch % 100 == 0 :#and epoch >= 800:
                #     torch.save(self.model, f'{self.MLDir}/model/{self.reg}/{self.task}/out/model_{self.job}_off{self.channels_off}_on{self.channels_on}_epoch_{epoch}_{self.train_size}.pt')
            
                #overwrite epochs where test loss are the minimum and mark in plot below:
                if epoch == 0:
                    min_loss = (test_loss/len(test_loader_in))
                    min_epoch = epoch
                elif (test_loss/len(test_loader_in)) < min_loss:
                    min_loss = (test_loss/len(test_loader_in))
                    min_epoch = epoch
                    torch.save(self.model, f'{self.MLDir}/model/{self.reg}/{self.task}/out/model_{self.job}_off{self.channels_off}_on{self.channels_on}_minepoch_{self.train_size}_fold_{k}.pt')
                
                #at last epoch
                if epoch == self.nepochs-1:
                    print(f'epoch:{epoch},training loss:{avg_train_ls:.5f},test loss:{avg_test_ls:.5f}', end='\r')
                    torch.save(self.model, f'{self.MLDir}/model/{self.reg}/{self.task}/out/model_{self.job}_off{self.channels_off}_on{self.channels_on}_epoch_{epoch}_{self.train_size}_fold_{k}.pt')

                #early stopping
                if epoch - min_epoch > self.es_gap:
                    print('early stopping at epoch:',epoch, 'min loss:',min_loss,'fold:',k)
                    # torch.save(self.model, f'{self.MLDir}/model/{self.reg}/{self.task}/out/model_{self.job}_off{self.channels_off}_on{self.channels_on}_estop_{epoch}_{self.train_size}.pt')
                    # ex.log_scalar(f'es_epoch_{self.job}/', min_epoch)
                    break
            
            self.min_epoch = min_epoch
            self.min_loss = min_loss
            print('min loss at epoch:',self.min_epoch, 'min loss:',self.min_loss,'fold:',k)
            #save model as artifact
            ex.add_artifact(filename=f'{self.MLDir}/model/{self.reg}/{self.task}/out/model_{self.job}_off{self.channels_off}_on{self.channels_on}_minepoch_{self.train_size}_fold_{k}.pt')

            #plot and save loss as png and npy
            self.plot_save_loss(k)
            np.save(f'{self.MLDir}/model/{self.reg}/{self.task}/out/train_loss_{self.job}_{self.train_size}_fold_{k}.npy', self.train_epoch_losses)
            np.save(f'{self.MLDir}/model/{self.reg}/{self.task}/out/test_loss_{self.job}_{self.train_size}_fold_{k}.npy', self.test_epoch_losses)

            #logging
            ex.log_scalar(f'min_loss_{self.job}/', self.min_loss)
            ex.log_scalar(f'min_epoch_{self.job}/', self.min_epoch) 
    
    #evaluation of the model happens here
    @ex.capture   
    def evaluateEDerror(self,
                    job, #job name
                    data_in, #training data offshore
                    data_deformfull, #training data deformation
                    data_out, #training data onshore
                    folds, #number of folds, also sets the split between train and test
                    channels_off = [64,128,256], #channels for offshore(1DCNN)
                    channels_on = [64,64], #channels for onshore(fully connected)
                    epoch =  None,#selected epoch
                    batch_size = 1000, #depends on GPU memory
                    control_points = [], #control points for evaluation
                    threshold = 0.1, #threshold for evaluation
                    # device =  torch.device("cpu"), #default for cpu incase running on non gpu node
                    reg_gaugeno = None,
                    ):
        
        self.job = job
        self.batch_size = batch_size
        self.channels_off = channels_off
        self.channels_on = channels_on
        self.folds = folds
        # self.device = device

        #read event list   
        event_list_path = f'{self.MLDir}/data/events/shuffled_events_test_{self.reg}_{self.test_size}.txt'
        event_list = np.loadtxt(event_list_path, dtype='str')

        #################PREDICTION CHUNK#####################

        #load dataloaders
        test_loader_in = self.dataloader(data_in)
        test_loader_deformfull = self.dataloader(data_deformfull)
        test_loader_out = self.dataloader(data_out)

        #define shape for data_out
        predic = np.zeros(data_out.shape) #mean
        residual = np.zeros(data_out.shape) #std
        mc_samples = []

        # Test model
        test_loss = 0
        test_residual_loss = 0
        
        for batch_idx, (batch_data_in, batch_data_deformfull, batch_data_out) in enumerate(zip(test_loader_in, test_loader_deformfull, test_loader_out)):
            
            batch_data_in = batch_data_in[0].to(self.device)
            batch_data_deformfull = batch_data_deformfull[0].to(self.device)
            batch_data_out = batch_data_out[0].to(self.device)

            if self.job == 'mc_dropout':
                recon_data, residual_data, point_samples = self.mc_dropout_inference(
                    wave_input=batch_data_in,
                    deform_input=batch_data_deformfull,
                    num_samples=100
                )
            else:
                print('Not using MC dropout model. Exiting.')
                sys.exit()

            # Move outputs to CPU for saving
            mc_samples.append(point_samples.cpu().numpy())
            predic[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size] = recon_data.cpu().numpy()
            residual[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size] = residual_data.cpu().numpy()

            # Compute test loss
            loss = self.criterion(recon_data, batch_data_out)
            test_loss += loss.item()

            true_residual = (batch_data_out - recon_data) ** 2
            residual_loss = self.criterion(residual_data, true_residual)
            test_residual_loss += residual_loss.item()

            print(f"Batch {batch_idx+1}/{len(test_loader_in)} — test loss: {test_loss / (batch_idx+1):.5f}, residual loss: {test_residual_loss / (batch_idx+1):.5f}")

        # Final MC samples shape: (samples, total_examples, locations)
        mc_samples = np.concatenate(mc_samples, axis=1)

        print(f"test loss: {test_loss / len(test_loader_in):.5f}")
        print(f"test residual loss: {test_residual_loss / len(test_loader_in):.5f}")

        #save prediction as numpy array
        np.save(f'{self.MLDir}/model/{self.reg}/{self.task}/out/pred_trainsize{self.train_size}_testsize{self.test_size}_direct.npy', predic)
        np.save(f'{self.MLDir}/model/{self.reg}/{self.task}/out/residual_trainsize{self.train_size}_testsize{self.test_size}_direct.npy', residual)
        np.save(f'{self.MLDir}/model/{self.reg}/{self.task}/out/mc_samples_trainsize{self.train_size}_testsize{self.test_size}_direct.npy', mc_samples)

        print('prediction done over all folds with size:',predic.shape)
        
        #################EVALUATION CHUNK#####################

        #first calculate location index of control points for given lat and lon
        locindices = get_idx_from_latlon(control_points)

        #evaluation table
        eve_perf = []
        true_list = []
        pred_list = []
        er_list = []
        res_list = []

        #score  returns mse_val,r2_val, truecopy[locindices],pred[locindices],pt_er, Gfit_val,l2n_val #,Kcap,Ksmall
        test_ids = np.loadtxt(f'{self.MLDir}/data/events/shuffled_events_test_{self.reg}_{self.test_size}.txt',dtype='str')
        for eve_no,eve in enumerate(test_ids):
            scores = calc_scores(data_out[eve_no,:],predic[eve_no,:],locindices,threshold)
            errors = calc_errors(residual[eve_no,:],locindices,var=False)
            eve_perf.append([scores[0],scores[1],scores[5],scores[6], #mse,r2,g,l2n
                            np.count_nonzero(data_out[eve_no,:]), #true count
                            np.count_nonzero(predic[eve_no,:]),#pred count
                            np.max(data_out[eve_no,:]), #true max
                            np.max(predic[eve_no,:]), #pred max
                            ]) 
            # mse_val,r2_val,pt_er,g_val,l2n_val,trueinundation,predinundation,
            true_list.append(scores[2])
            pred_list.append(scores[3])
            er_list.append(scores[4])
            res_list.append(errors)

        #count of events less than 
        eve_perf = np.array(eve_perf)
        true_list = np.array(true_list)
        pred_list = np.array(pred_list)
        er_list = np.array(er_list)
        res_list = np.array(res_list)

        #convert eve_perf to dataframe
        eve_perf = pd.DataFrame(eve_perf,columns=['mse','r2','g','l2n','true','pred','truemax','predmax'])
        eve_perf['id'] = event_list

        #save eve_perf as csv
        eve_perf.to_csv(f'{self.MLDir}/model/{self.reg}/{self.task}/out/model_direct_off{self.channels_off}_on{self.channels_on}_{self.train_size}_eve_perf_testsize{self.test_size}.csv')

        #combine columns of true,pred,er into 12 column array
        true_pred_er = np.column_stack((true_list,pred_list,er_list,res_list))

        #print overall performance metrics for whole training exercise and evaluation work : 
        # Plot results max height for all events
        test_max = np.max(data_out,axis=(1))
        recon_max = np.max(predic,axis=(1))

        # Calculate mseoverall and r2maxdepth
        r2maxdepth = r2_score(test_max, recon_max)
        r2area = r2_score(eve_perf['true'], eve_perf['pred'])
        mseoverall = mean_squared_error(data_out, predic)
        gfitoverall = np.mean(Gfit(data_out, predic))
        print(f"mseoverall: {mseoverall:.4f}")
        print(f"r2maxdepth: {r2maxdepth:.3f}")
        print(f"r2area: {r2area:.3f}")
        print(f"gfitoverallmean: {gfitoverall:.3f}")
        #TODO: add per event evaluation for discovery and analysis

        #log metrics to sacred
        ex.log_scalar('mseoverall',mseoverall)
        ex.log_scalar('r2maxdepth',r2maxdepth)
        ex.log_scalar('r2area',r2area)
        ex.log_scalar('gfitoverall',gfitoverall)

        #################PLOTTING CHUNK#####################
        
        # Create a single figure with three axes
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Plot scatter of flood count
        scatter = ax[0].scatter(eve_perf['true'], eve_perf['pred'], s=1, c=eve_perf['g'], cmap='PiYG')
        ax[0].plot([0, 1], [0, 1], transform=ax[0].transAxes, color='red')
        ax[0].set_title(f"r^2: {r2area:.3f} for inundated pixel above 0.2 m")
        plt.colorbar(scatter, ax=ax[0])
        ax[0].set_aspect('equal', adjustable='box')
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[0].set_xlim(10, 1000000)
        ax[0].set_ylim(10, 1000000)
        ax[0].grid()
        ax[0].set_xlabel('True: Inundated pixel count')
        ax[0].set_ylabel('Predicted: Inundated pixel count')

        # Plot scatter of mse vs l2norm
        scatter = ax[1].scatter(eve_perf['l2n'], eve_perf['mse'], s=1, c=eve_perf['g'], cmap='PiYG')
        ax[1].plot([0, 1], [0, 1], transform=ax[1].transAxes, color='red')
        plt.colorbar(scatter, ax=ax[1])
        ax[1].set_title(f"L2 Error vs L2 Norm for inundated pixel above 0.2 m")
        ax[1].set_aspect('equal', adjustable='box')
        ax[1].set_xscale('log')
        ax[1].set_yscale('log')
        ax[1].set_xlim(.00001, 10)
        ax[1].set_ylim(.00001, 10)
        ax[1].grid()
        ax[1].set_xlabel('L2 Norm')
        ax[1].set_ylabel('L2 Error')

        # Plot scatter of max depth for each event
        # Calculate the point density
        x = test_max
        y = recon_max
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        index = z.argsort()
        x, y, z = x[index], y[index], z[index]
        scatter = ax[2].scatter(x, y, c=z, s=1)
        ax[2].plot([0, 1], [0, 1], transform=ax[2].transAxes, color='red')
        plt.colorbar(scatter, ax=ax[2])
        ax[2].set_title("Max depth for each event")
        ax[2].text(10, 5, f"R Squared: {r2maxdepth:.5f} ", fontsize=12)
        ax[2].set_aspect('equal', adjustable='box')
        ax[2].set_xlim(0, 20)
        ax[2].set_ylim(0, 20)
        ax[2].grid()
        plt.xlabel('True:Max. inun. depth(m)')
        plt.ylabel('Predicted:Max. inun. depth(m)')

        # Adjust layout and add a suptitle to the main plot
        plt.tight_layout()
        plt.suptitle(f"mseoverall: {mseoverall:.5f}, r2maxdepth: {r2maxdepth:.5f}, gfitoverall: {gfitoverall:.4f}, testsize: {self.test_size}", fontsize=16, y=0.02)
        plt.savefig(f'{self.MLDir}/model/{self.reg}/{self.task}/plot/model_direct_off{self.channels_off}_on{self.channels_on}_{self.train_size}_maxdepth_testsize{self.test_size}.png')
        plt.clf()
        ex.add_artifact(filename=f'{self.MLDir}/model/{self.reg}/{self.task}/plot/model_direct_off{self.channels_off}_on{self.channels_on}_{self.train_size}_maxdepth_testsize{self.test_size}.png')        

        #plot error at each location
        print('plotting error at each control points')
        plt.figure(figsize=(15, 30))

        #add to main plot the mse and r2 to the plot at the top
        plt.suptitle(f"mseoverall: {mseoverall:.5f},r2maxdepth: {r2maxdepth:.5f}gfitoverall:{gfitoverall:.4f},testsize: {self.test_size}",fontsize=25)
               
        #error charts
        plt.figure(figsize=(25, 30))
        plt.suptitle(f"mseoverall: {mseoverall:.5f},r2maxdepth: {r2maxdepth:.5f}gfitoverall:{gfitoverall:.4f},testsize: {self.test_size}",fontsize=25)
        for i in range(len(locindices)):
            ax = plt.subplot(4, 3, i + 1)   
            # Plot the histogram of errors for the control locations
            plt.hist(er_list[er_list[:,i]!=0,i],bins=40,edgecolor='black',)
            # Check if its empty
            if len(er_list[er_list[:,i]!=0,i]) == 0:
                continue
            else:
                quantiles = np.percentile(er_list[er_list[:,i]!=0,i], [5, 50, 95])
                # Plot quantile lines
                for q in quantiles:
                    ax.axvline(q, color='red', linestyle='--', label=f'Q{int(q)}')
                ax.set_xlim(-3, 3)
            #calculate hit and mis for each location based on depth of true and prediction
            #events crossing the threshold say 0.2 are considered flooded
            neve = np.count_nonzero(true_pred_er[:,i]>threshold)
            neve_recon = np.count_nonzero(true_pred_er[:,i+len(locindices)]>threshold)
            print(f"Control Location:{i+1},No of flood events:{neve}/{len(true_pred_er[:,i])}")
            if neve == 0:
                TP = -999
                FN = -999
            else:
            #true positive: true>0.2 and pred>0.2
                TP = np.count_nonzero((true_pred_er[:,i]>threshold) & (true_pred_er[:,i+len(locindices)]>threshold))/(neve)
                FN = np.count_nonzero((true_pred_er[:,i]>threshold) & (true_pred_er[:,i+len(locindices)]<=threshold))/(neve)
            TN = np.count_nonzero((true_pred_er[:,i]<=threshold) & (true_pred_er[:,i+len(locindices)]<=threshold))/(len(true_pred_er[:,i])-neve)
            FP = np.count_nonzero((true_pred_er[:,i]<=threshold) & (true_pred_er[:,i+len(locindices)]>threshold))/(len(true_pred_er[:,i])-neve)
            plt.title(f"Control Location:{i+1},No of flood events:True#{neve}|Predicted#{neve_recon}",fontsize=15)
            plt.text(0.8, 0.9, f" TP: {TP:.2f}, TN: {TN:.2f}", horizontalalignment='center',verticalalignment='center', transform=plt.gca().transAxes,fontsize=15)
            plt.text(0.8, 0.75, f"FP: {FP:.2f}, FN: {FN:.2f}", horizontalalignment='center',verticalalignment='center', transform=plt.gca().transAxes,fontsize=15)
            plt.xlabel('Error in max flood depth (m)',fontsize=15)
            plt.ylabel('Count',fontsize=15)

            # Create a new inset axis for the scatter plot
            axins = inset_axes(ax, width="30%", height="30%", loc='upper left', borderpad=3)
            
            # Scatter plot of values (replace with your data)
            axins.plot([0, 1], [0, 1], transform=axins.transAxes, color='blue',)
            axins.scatter(true_pred_er[:,i], true_pred_er[:,i+len(locindices)], marker='o', color='red', label='Max Inun Depth',s=0.33)  # Customize marker and color as needed
            axins.text(0.5,5,f'r^2:{r2_score(true_pred_er[:,i], true_pred_er[:,i+len(locindices)]):.2f}',fontsize=15)
            axins.set_xlim(0, 6)  # Adjust x-axis limits for the scatter plot
            axins.set_ylim(0, 6)  # Adjust y-axis limits for the scatter plot
            axins.set_xlabel('True')
            axins.set_ylabel('Predicted')
            axins.set_aspect('equal', adjustable='box')
        plt.savefig(f'{self.MLDir}/model/{self.reg}/{self.task}/plot/model_direct_off{self.channels_off}_on{self.channels_on}_{self.train_size}_error_testsize{self.test_size}.png')
        plt.clf()
        ex.add_artifact(f'{self.MLDir}/model/{self.reg}/{self.task}/plot/model_direct_off{self.channels_off}_on{self.channels_on}_{self.train_size}_error_testsize{self.test_size}.png')

        #plot box plot for event performance using metric G
        table = eve_perf

        # Read sampling file and merge with 'table' based on 'id'
        sampling_file = pd.read_csv(f'{self.MLDir}/data/info/sampling_input_{self.reg}_{reg_gaugeno}.csv')
        table = table.merge(sampling_file, on='id', how='left')

        # Set the bin edges for 'max_off' values
        bin_edges = [0, 0.1, 0.3, 1, 2, 3, 99]  # Define your custom bin edges here

        # Create a box plot of 'gfit_out' binned by 'max_off' using custom bin edges
        plt.figure(figsize=(8, 4))
        table['max_off_bin'] = pd.cut(table['max_off'], bin_edges)
        table.boxplot(column='g', by='max_off_bin', vert=True, showfliers=True, widths=0.2, sym='k.', patch_artist=True, ax=plt.gca())

        # Get bin counts
        bin_counts = table.groupby('max_off_bin').size()

        # Set labels and titles
        plt.ylabel('Goodness of fit (G)', fontsize=15)
        plt.xlabel('Maximum offshore amplitude (m)', fontsize=15)
        plt.text(4, 0.3, f'Event Count by {bin_counts}', color='red')
        plt.title(f'G vs. max_off for {self.reg}', fontsize=15)
        plt.savefig(f'{self.MLDir}/model/{self.reg}/{self.task}/plot/model_direct_off{channels_off}_on{channels_on}_{self.train_size}_gfit_testsize{self.test_size}.png')
        plt.clf()
        ex.add_artifact(f'{self.MLDir}/model/{self.reg}/{self.task}/plot/model_direct_off{channels_off}_on{channels_on}_{self.train_size}_gfit_testsize{self.test_size}.png')
        
        #################TABULATE CHUNK#####################
        
        #save table
        table.to_csv(f'{self.MLDir}/model/{self.reg}/{self.task}/out/model_direct_off{self.channels_off}_on{self.channels_on}_{self.train_size}_compile_testsize{self.test_size}.csv')

        #add column names as T1...P1...and E1...
        col_names = []
        for i in range(len(control_points)):
            col_names.append(f'T{i+1}')
        for i in range(len(control_points)):
            col_names.append(f'P{i+1}')
        for i in range(len(control_points)):
            col_names.append(f'E{i+1}')
        for i in range(len(control_points)):
            col_names.append(f'R{i+1}')
            
        #convert to dataframe
        true_pred_er = pd.DataFrame(true_pred_er,columns=col_names)
        true_pred_er['id'] = event_list

        #save true_pred_er as csv
        true_pred_er.to_csv(f'{self.MLDir}/model/{self.reg}/{self.task}/out/model_direct_off{self.channels_off}_on{self.channels_on}_{self.train_size}_true_pred_er_testsize{self.test_size}.csv')

