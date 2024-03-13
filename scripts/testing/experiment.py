#experiment.py
#This is where the model is built with config parameters and the ML train and evaluation ie the experiment is designed
import os
import numpy as np
import pandas as pd
import scipy.signal
import xarray as xr

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib.pyplot as plt
import contextily as cx

import neptune
from neptune.integrations.sacred import NeptuneObserver
from sacred import Experiment
from sacred.observers import FileStorageObserver

# # # Set up Sacred experiment
ex = Experiment("autoencoder_experiment")

# Set up Sacred observers, Set up the neptune instance and logger. Turn off if using just some function from this file
ex.observers.append(FileStorageObserver.create("sacred_logs"))

run = neptune.init_run(project="naveenragur/ML4Sicily",
                    source_files=["experiment.py","main.py","parameters.json","run.sbatch"],
                    api_token=os.getenv('Neptune_api_token'),
                    )
ex.observers.append(NeptuneObserver(run=run))

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
    train_size = "2500" #eventset size for training & building the model
    mask_size = "6317" #eventset size for masking
    test_size = "6421" #eventset size for testing 

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
    lr_on = 0.0025
    lr_deform = 0.0025
    lr_couple = 0.01

    es_gap = 500
    step_size = 300
    gamma = 0.9
    
    batch_size = 300
    batch_size_on = 300
    batch_size_deform = 300
    nepochs = 3500
    split = 0.75

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
    tune_nlayers = 0 #last n layer of encoder and first layer of decoder are also tunable

    #for reading data
    windowthreshold = 0.1
    twindow = 480 #in mins

    #for evaluation
    threshold = 0.2 #threshold for flooded or not in evaluation

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

#build the offshore model
class Autoencoder_offshore(nn.Module):
    def __init__(self,
                 ninputs=5, #number of input channels or gauges
                 t_len = 480, #number of time steps
                 ch_list = [32,64,96], #number of channels in each layer,
                 zdim = 50,
                 hdim = 50):#number of latent variables
        
        super(Autoencoder_offshore, self).__init__()
        # more channels mean more fine details, more resolution 
        # less channel and layers less likely to overfit so better maximas and minimas
        # more accuracy but slower and more memory and data needed to train
        self.ch_list = ch_list
        self.hdim = hdim
        self.zdim = zdim
        
        # define encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(ninputs, ch_list[0], kernel_size=3, padding=1),   
            nn.LeakyReLU(negative_slope=0.5,inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(ch_list[0], ch_list[1], kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.5,inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(ch_list[1], ch_list[2], kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.5,inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # nn.Dropout(0.1),
        ) 

        # define decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(ch_list[2], ch_list[1], kernel_size=4, padding=1, stride= 2),  
            nn.LeakyReLU(negative_slope=0.5,inplace=True) ,
            nn.ConvTranspose1d(ch_list[1], ch_list[0], kernel_size=4, padding=1, stride= 2), 
            nn.LeakyReLU(negative_slope=0.5,inplace=True) ,
            nn.ConvTranspose1d(ch_list[0], ninputs, kernel_size=4, padding=1, stride= 2), 
            nn.LeakyReLU(negative_slope=0.5,inplace=True) ,
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(zdim),
        )

        self.fc2 = nn.Sequential(
            nn.LazyLinear((t_len*ch_list[2])//int(2**(len(ch_list)))),
            nn.Unflatten(1,(ch_list[2],int(t_len//2**(len(ch_list))))),
        )

        # self.fc2 = nn.Sequential(
        #     nn.LazyLinear((t_len*hdim)//int(2**(len(ch_list)))),
        #     nn.Unflatten(1,(int(t_len//2**(len(ch_list))),hdim)),
        # )

        # Add the LSTM layer separately outside of the Sequential module which can be retuned during coupling
        # self.lstm_encoder = nn.Sequential(
        #     nn.LSTM(input_size=ch_list[2], hidden_size=hdim,num_layers=1, batch_first=True),
        # )

        # self.lstm_decoder = nn.Sequential(
        #     nn.LSTM(input_size=hdim, hidden_size=ch_list[2],num_layers=1, batch_first=True),
        # )


    def encode(self, x):
        x = self.encoder(x) #cnn encoder
        # x = x.permute(0, 2, 1)#permute to (batch, seq-t, features-gauges or channels)
        # x, (h_n, c_n) = self.lstm_encoder(x)
        return x
    
    def decode(self, x):
        # x, (h_n, c_n) = self.lstm_decoder(x)
        # x = x.permute(0, 2, 1)#permute to (batch, seq-t, features-gauges or channels)
        x = self.decoder(x) #cnn decoder
        return x
    
    def forward(self, x):
        # x = nn.functional.interpolate(x, scale_factor=4, mode='linear')
        #reverse data[:,:,:] on second axis
        x = self.encode(x)  
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.decode(x)      
        # x = nn.functional.interpolate(x, scale_factor=0.25, mode='linear')
        return x

#build the onshore model
class Autoencoder_onshore(nn.Module):
    def __init__(self,
                 xy, #number of input channels or grids for flooding
                 zlist = [32,64,128]): #number of channels in each layer
        super(Autoencoder_onshore, self).__init__()
        # more channels mean more fine details, more resolution 
        # less channel and layers less likely to overfit so better maximas and minimas
        # more accuracy but slower and more memory and data needed to train
        self.xy = xy
        
        # define encoder layers
        if len(zlist) == 1:
            self.encoder = nn.Sequential(          
                nn.Linear(xy, zlist[0]),
                nn.LeakyReLU(negative_slope=0.01,inplace=True),
            )
            self.decoder = nn.Sequential(
                nn.Linear(zlist[0], xy),
                nn.LeakyReLU(negative_slope=0.01,inplace=True),
            )
        elif len(zlist) == 2:
            self.encoder = nn.Sequential(          
                nn.Linear(xy, zlist[0]),
                nn.LeakyReLU(negative_slope=0.01,inplace=True),
                nn.Linear(zlist[0], zlist[1]),
                nn.LeakyReLU(negative_slope=0.01,inplace=True),
            )
            self.decoder = nn.Sequential(
                nn.Linear(zlist[1], zlist[0]),
                nn.LeakyReLU(negative_slope=0.01,inplace=True),
                nn.Linear(zlist[0], xy),
                nn.LeakyReLU(negative_slope=0.01,inplace=True),
            )
        elif len(zlist) == 3:
            self.encoder = nn.Sequential(          
                nn.Linear(xy, zlist[0]),
                nn.LeakyReLU(negative_slope=0.01,inplace=True),
                nn.Linear(zlist[0], zlist[1]),
                nn.LeakyReLU(negative_slope=0.01,inplace=True),
                nn.Linear(zlist[1], zlist[2]),
                nn.LeakyReLU(negative_slope=0.01,inplace=True),
            )
            self.decoder = nn.Sequential(
                nn.Linear(zlist[2], zlist[1]),
                nn.LeakyReLU(negative_slope=0.01,inplace=True),
                nn.Linear(zlist[1], zlist[0]),
                nn.LeakyReLU(negative_slope=0.01,inplace=True),
                nn.Linear(zlist[0], xy),
                nn.LeakyReLU(negative_slope=0.01,inplace=True),
            )

    def encode(self, x):
        x = self.encoder(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        x = self.encode(x)  
        x = self.decode(x)
        return x

#build the onshore model
class AutoencoderSplitOnshore(nn.Module):
    def __init__(self, xy, parts, zlist=[16, 256, 256]):
        super(AutoencoderSplitOnshore, self).__init__()

        # Split the input into 'parts' 
        self.xy = xy
        self.split_size = (xy // parts)+1
        self.parts = parts

        # Encoder and Decoder for each split
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(parts):
            encoder = nn.Sequential(
                nn.LazyLinear( zlist[0]),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            )
            self.encoders.append(encoder)

            decoder = nn.Sequential(
                nn.LazyLinear(self.split_size),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            )
            self.decoders.append(decoder)

        # Global encoder and decoder
        self.global_encoder = nn.Sequential(
            nn.LazyLinear(zlist[1]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.LazyLinear(zlist[2]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.global_decoder = nn.Sequential(
            nn.LazyLinear(zlist[-2]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.LazyLinear(zlist[-1]),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.LazyLinear(zlist[0]*parts),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        # print(x.shape)
        # print(self.split_size)

        # Split into 'parts' 
        split_x = torch.chunk(x, self.parts, dim=1)
        # print(len(split_x))
        # print(split_x[0].shape)

        # Encode each split independently #600k/ parts to z[0]*parts 
        encoded_splits = []
        for i in range(self.parts):  
            encoded_split = self.encoders[i](split_x[i])
            encoded_splits.append(encoded_split)
        # print(len(encoded_splits))
        # print(encoded_splits[0].shape)

        # Concatenate the encoded parts
        concat_splitencoding = torch.cat(encoded_splits, dim=1) #z[0]*parts 16*64=1024
        # print(concat_splitencoding.shape)

        # Global encoding
        global_encoded = self.global_encoder(concat_splitencoding) #z[0]*parts to z[-1] 1024->128->128
        # print(global_encoded.shape)

        # Global decoding
        global_decoded = self.global_decoder(global_encoded) #z[-1] to z[0]*parts 128->128->1024
        # print(global_decoded.shape)

        # Split the combined decoded output
        split_global_decoded = torch.chunk(global_decoded, self.parts, dim=1) #z[0]*parts  1024/64=16
        # print(len(split_global_decoded))
        # print(split_global_decoded[0].shape)

        # Decode each split independently #z[0]*parts to 600k/parts
        decoded_splits = []
        for i in range(self.parts): #z[0]*parts to 600k/parts
            decoded_split = self.decoders[i](split_global_decoded[i])
            decoded_splits.append(decoded_split)
        # print(len(decoded_splits))
        # print(decoded_splits[0].shape)

        # Concatenate the decoded parts
        reconstructed_x = torch.cat(decoded_splits, dim=1) #600k/parts to 600k
        print(reconstructed_x.shape)
        reconstructed_x = reconstructed_x[:, :self.xy] #crop to original size

        return reconstructed_x
    
#build the deform model
class Autoencoder_deformation(nn.Module):
    def __init__(self,
                 xy,
                 df_list=[16, 32, 64, 128], zdim=50):
        super(Autoencoder_deformation, self).__init__()
        self.xy = xy

        # Define encoder layers with 1D convolution and varying kernel sizes and strides
        self.encoder = nn.Sequential(
            nn.Conv1d(1, df_list[0], kernel_size=7, stride=4, padding=1),
            nn.LeakyReLU(negative_slope=0.5, inplace=True),
            nn.MaxPool1d(kernel_size=10, stride=10),
            nn.Conv1d(df_list[0], df_list[1], kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.5, inplace=True),
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(df_list[1], df_list[2], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.5, inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(df_list[2], df_list[3], kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.5, inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(0.1),
        )

        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(df_list[3], df_list[2], kernel_size=7, stride=4, padding=1),
            nn.LeakyReLU(negative_slope=0.5, inplace=True),
            nn.ConvTranspose1d(df_list[2], df_list[1], kernel_size=6, stride=4, padding=1),
            nn.LeakyReLU(negative_slope=0.5, inplace=True),
            nn.ConvTranspose1d(df_list[1], df_list[0], kernel_size=5, stride=4, padding=1),
            nn.LeakyReLU(negative_slope=0.5, inplace=True),
            nn.ConvTranspose1d(df_list[0], 1, kernel_size=5, stride=4, padding=1),
            nn.LeakyReLU(negative_slope=0.5, inplace=True),
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(zdim),
            # nn.Dropout(0.1),
        )

        self.fc2 = nn.Sequential(
            nn.LazyLinear(df_list[3] * (xy // 256)),
            # nn.Dropout(0.1),
            nn.Unflatten(1,(df_list[3],(xy // 256))),
        )


    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x

    def forward(self, x):
        x = x.unsqueeze(1) #add channel dimension
        x = self.encode(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.decode(x)
        # print(x.shape)
        x = x[:, :, :self.xy] #crop to original size
        x = x.squeeze(1) #remove channel dimension
        return x

# #build the deform model
# class Autoencoder_deformationf(nn.Module):
#     def __init__(self,
#                  x,
#                  y,
#                  df_list=[16, 32, 64, 128], zdim=50):
#         super(Autoencoder_deformationf, self).__init__()
#         self.x = x
#         self.y = y

#         # Define encoder layers with 1D convolution and varying kernel sizes and strides
#         self.encoder = nn.Sequential(
#             nn.MaxPool1d(kernel_size=5, stride=5),
#             nn.Conv1d(1, df_list[0], kernel_size=5, stride=4, padding=1),
#             nn.LeakyReLU(negative_slope=0.5, inplace=True),
#             nn.MaxPool1d(kernel_size=5, stride=5),
#             nn.Conv1d(df_list[0], df_list[1], kernel_size=5, stride=2, padding=1),
#             nn.LeakyReLU(negative_slope=0.5, inplace=True),
#             nn.MaxPool1d(kernel_size=5, stride=5),
#             nn.Conv1d(df_list[1], df_list[2], kernel_size=3, stride=2, padding=1),
#             nn.LeakyReLU(negative_slope=0.5, inplace=True),
#             nn.MaxPool1d(kernel_size=3, stride=3),
#             nn.Conv1d(df_list[2], df_list[3], kernel_size=3, stride=2, padding=1),
#             nn.LeakyReLU(negative_slope=0.5, inplace=True),
#             nn.MaxPool1d(kernel_size=3, stride=3),
#             nn.Dropout(0.1),
#         )
#         # Define decoder layers
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose1d(df_list[3], df_list[2], kernel_size=3, stride=2, padding=1),
#             nn.LeakyReLU(negative_slope=0.5, inplace=True),
#             nn.Upsample(scale_factor=2, mode='linear'),
#             nn.ConvTranspose1d(df_list[2], df_list[1], kernel_size=3, stride=2, padding=1),
#             nn.LeakyReLU(negative_slope=0.5, inplace=True),
#             nn.Upsample(scale_factor=3, mode='linear'),
#             nn.ConvTranspose1d(df_list[1], df_list[0], kernel_size=3, stride=2, padding=1),
#             nn.LeakyReLU(negative_slope=0.5, inplace=True),
#             nn.Upsample(scale_factor=2, mode='linear'),
#             nn.ConvTranspose1d(df_list[0], 1, kernel_size=3, stride=2, padding=1),
#             nn.LeakyReLU(negative_slope=0.5, inplace=True),
#             nn.Upsample(scale_factor=20, mode='linear'),
#         )

#         self.fc1 = nn.Sequential(
#             nn.Flatten(),
#             nn.LazyLinear(zdim),
#             # nn.Dropout(0.1),
#         )

#         self.fc2 = nn.Sequential(
#             nn.LazyLinear(df_list[3] * (x*y // 3800)),
#             # nn.Dropout(0.1),
#             nn.Unflatten(1,(df_list[3],(x*y // 3800))),
#         )


#     def encode(self, x):
#         x = self.encoder(x)
#         return x

#     def decode(self, x):
#         x = self.decoder(x)
#         return x

#     def forward(self, x):
#         x = x.unsqueeze(1) #add channel dimension #batch x xdim x ydim -> batch x 1 x xdim x ydim
#         x = x.flatten(2) #flatten to batch x 1 x (xdim x ydim)
#         #flatten to batch x 1 x (xdim x ydim)
#         x = self.encode(x)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.decode(x)
#         # print(x.shape)
#         x = x[:, :, :self.x*self.y] #crop to original size
#         x = x.reshape(x.shape[0],1,self.y,self.x) #back to batch x 1 x xdim x ydim    
#         x = x.squeeze(1) #remove channel dimension
#         return x

class Autoencoder_deformationf(nn.Module):
    def __init__(self,
                 x,
                 y,
                 df_list=[16, 32, 64, 128], zdim=50):
        super(Autoencoder_deformationf, self).__init__()
        self.x = x
        self.y = y

        # Define encoder layers with 1D convolution and varying kernel sizes and strides
        self.encoder = nn.Sequential(
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
        # Define decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(df_list[3], df_list[2], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.5, inplace=True),
            nn.ConvTranspose2d(df_list[2], df_list[1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.5, inplace=True),
            nn.ConvTranspose2d(df_list[1], df_list[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.5, inplace=True),
            nn.ConvTranspose2d(df_list[0], 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(negative_slope=0.5, inplace=True),
            nn.Upsample(size=[y,x], mode='bilinear'),
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(zdim),
            # nn.Dropout(0.1),
        )

        self.fc2 = nn.Sequential(
            nn.LazyLinear(df_list[3] * 240),
            # nn.Dropout(0.1),
            nn.Unflatten(1,(df_list[3],24,10)),
        )

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        return x

    def forward(self, x):
        x = x.unsqueeze(1) #add channel dimension #batch x xdim x ydim -> batch x 1 x xdim x ydim
        x = self.encode(x)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        # print(x.shape)
        x = self.decode(x)
        # print(x.shape)
        x = x.squeeze(1) #remove channel dimension #batch x 1 x xdim x ydim -> batch x xdim x ydim
        return x

#build the coupled model
class Autoencoder_coupled(nn.Module):
    def __init__(self,
                offshore_model,
                onshore_model,
                deform_model,
                interface_layers,
                tune_nlayers):
        super(Autoencoder_coupled, self).__init__()

        # Pretrained offshore 
        self.offshore_encoder = offshore_model.encoder
        for i, layer in enumerate(self.offshore_encoder):
                for param in layer.parameters():
                    param.requires_grad = False

        self.offshore_fc1 = offshore_model.fc1
        for i, layer in enumerate(self.offshore_fc1):
                for param in layer.parameters(): #layer is tunable
                    param.requires_grad = True

        #some fine tuning of offshore model,TODO:convert to LSTM later if needed
        self.offshore_tune = nn.Sequential(
                                nn.LazyLinear(128),
                                nn.LeakyReLU(inplace=True),
            )

        # Pretrained deform 
        # self.deform_encoder = deform_model.encoder
        # for i, layer in enumerate(self.deform_encoder):
        #         for param in layer.parameters():
        #             param.requires_grad = False
        # self.deform_fc1 = deform_model.fc1
        # for i, layer in enumerate(self.offshore_fc1):
        #         for param in layer.parameters(): #layer is tunable
        #             param.requires_grad = False

        #some fine tuning of deform model
        # self.deform_tune = nn.Sequential(
        #                         nn.Linear(
        #                             in_features=64, out_features=64
        #                         ),
        #                         nn.LeakyReLU(inplace=True),
        #     ) 

        # Pretrained onshore model
        self.onshore_decoder = onshore_model.decoder 
        for i, layer in enumerate(self.onshore_decoder):
            if i < tune_nlayers:
                for param in layer.parameters(): #first layer is tunable
                    param.requires_grad = True
            else:
                for param in layer.parameters(): #all layers except first layer are frozen
                    param.requires_grad = True

        # Interface
        self.interface_layers = interface_layers
        if self.interface_layers == 1:
            self.connect = nn.Sequential(
                                nn.LazyLinear(128),
                                nn.LeakyReLU(inplace=True),
            ) 
        elif self.interface_layers == 2:    
            self.connect = nn.Sequential(
                                nn.LazyLinear(128),
                                nn.LeakyReLU(inplace=True),
                                nn.LazyLinear(128),
                                nn.LeakyReLU(inplace=True),
            )

        # #remap filtered onshore depth corrections before applying
        # self.remap = nn.Sequential(
        #                         nn.LazyLinear(4),
        #                         nn.LeakyReLU(inplace=True),
        #                         nn.LazyLinear(594725),
        #                         nn.LeakyReLU(inplace=True),
        #     ) 


    def forward(self, x, dz):
        #encode offshore time series to latent space
        x = self.offshore_encoder(x)  #3 layers of CNN
        x = self.offshore_fc1(x) #flatten and linear layer 64 latent space
        x = self.offshore_tune(x) #retune to another 64 latent space or a different latent space
        if self.interface_layers > 0: #linear layer 
            z = self.connect(x)
        y = self.onshore_decoder(z)
        return y  
      
        #encode deformation to latent space
        #make copy of dz for skip connection
        # dz_raw = dz 
        # dz = dz.unsqueeze(1)
        # dz = self.deform_encoder(dz) #4 layers of CNN
        # dz = self.deform_fc1(dz) #flatten and linear layer 64 latent space
        # dz = self.deform_tune(dz) #retune to another 64 latent space  or a different latent space

        # #common latent space from from encoders
        # z = torch.cat((x, dz), dim=1) #with concat converts to 128 latent space
           
        #retuning to onshore latent space
        # if self.interface_layers > 0: #linear layer 
        #     z = self.connect(x)
            # deltaz = self.connect(z)
            # z = torch.add(z,deltaz)
        
        #decode to onshore
        # y = self.onshore_decoder(z)
        # return y
        # dz_x_y = torch.mul(dz_raw,y)
        # dy = self.remap(dz_x_y)
        # corrected_y = torch.add(y,dy)
        # return corrected_y
    
#build the coupled model
class Autoencoder_coupled2(nn.Module):
    def __init__(self,
                offshore_model,
                onshore_model,
                deform_model,
                interface_layers,
                tune_nlayers,
                parts,
                xy):
        super(Autoencoder_coupled2, self).__init__()

        self.parts = parts
        self.xy = xy

        # Pretrained offshore 
        self.offshore_encoder = offshore_model.encoder
        for i, layer in enumerate(self.offshore_encoder): #first tune_n layers are frozen and rest free
            if i  <= (len(self.offshore_encoder ) - tune_nlayers):
                for param in layer.parameters(): #first layers are frozen
                    param.requires_grad = False
            else:
                for param in layer.parameters(): #last layers are tunable
                    param.requires_grad = True

        self.offshore_fc1 = offshore_model.fc1
        for i, layer in enumerate(self.offshore_fc1):
                for param in layer.parameters(): #FC layer is tunable
                    param.requires_grad = True

        #some fine tuning of offshore model,
        #TODO:convert to LSTM later if needed
        self.offshore_tune = nn.Sequential(
                                nn.LazyLinear(64),
                                nn.LeakyReLU(inplace=True),
            )

        # Pretrained deform 
        self.deform_encoder = deform_model.encoder
        if i  <= (len(self.deform_encoder ) - tune_nlayers):
            for param in layer.parameters(): #first layers are frozen
                param.requires_grad = False
        else:
            for param in layer.parameters(): #last layers are tunable
                param.requires_grad = True
        
        self.deform_fc1 = deform_model.fc1
        for i, layer in enumerate(self.deform_fc1):
                for param in layer.parameters(): # last layer is tunable
                    param.requires_grad = True

        # some fine tuning of deform model
        self.deform_tune = nn.Sequential(
                                nn.LazyLinear(64),
                                nn.LeakyReLU(inplace=True),
            ) 

        # Pretrained onshore model
        self.onshore_global_decoder = onshore_model.global_decoder 
        for i, layer in enumerate(self.onshore_global_decoder):
            if i <= tune_nlayers:
                for param in layer.parameters(): #first layer is tunable
                    param.requires_grad = True
            else:
                for param in layer.parameters(): #second layer is frozen
                    param.requires_grad = False


        self.onshore_split_decoders = onshore_model.decoders
        #freeze all decoder layers in the onshore_split_decoders 
        for i, decoder in enumerate(self.onshore_split_decoders):
            for param in decoder.parameters():
                param.requires_grad = False

        # Interface
        self.interface_layers = interface_layers
        if self.interface_layers == 1:
            self.connect = nn.Sequential(
                                nn.LazyLinear(128),
                                nn.LeakyReLU(inplace=True),
            ) 
        elif self.interface_layers == 2:    
            self.connect = nn.Sequential(
                                nn.LazyLinear(128),
                                nn.LeakyReLU(inplace=True),
                                nn.LazyLinear(128),
                                nn.LeakyReLU(inplace=True),
            )

        #remap filtered onshore depth corrections before applying
        # self.remap = nn.Sequential(
        #                         nn.LazyLinear(4),
        #                         nn.LeakyReLU(inplace=True),
        #                         nn.LazyLinear(self.xy),
        #                         nn.LeakyReLU(inplace=True),
        #     ) 

        self.dropout = nn.Dropout(0.5)
        self.ht = nn.LeakyReLU(negative_slope=0.5, inplace=True)

    def forward(self, x, dz_red, dz_raw):

        #encode offshore time series to latent space
        x = self.offshore_encoder(x)  #3 layers of CNN
        x = self.offshore_fc1(x) #flatten and linear layer 64 latent space
        # x = self.offshore_tune(x) #retune to another 64 latent space or a different latent space

        # #encode deformation to latent space
        dz = dz_raw.unsqueeze(1) #add channel dimension
        dz = self.deform_encoder(dz) #4 layers of CNN
        dz = self.deform_fc1(dz) #flatten and linear layer 64 latent space
        dz = dz.squeeze(1) #remove channel dimension
        # dz = self.ht(dz)
        # dz = self.deform_tune(dz) #retune to another 64 latent space  or a different latent space
        # dz = self.ht(dz)

        # #common latent space from from encoders
        z = torch.cat((x, dz), dim=1) #with concat converts to 128 latent spacez
        # z = torch.add(x,dz)
        z = self.dropout(z) #lets say all the latent space is not useful for inundaiton prediction
           
        #retuning to transform onshore latent space
        if self.interface_layers > 0: #linear layer 
            z = self.connect(x)
            # deltaz = self.connect(z)
            # z = torch.add(z,deltaz)
        
        #decode to onshore global intermediate latent space
        global_decoded = self.onshore_global_decoder(z) #16*64=1024

        # Split the combined decoded latent
        split_global_decoded = torch.chunk(global_decoded, self.parts, dim=1) #z[0]*parts 

        # Decode each split independently
        decoded_splits = []
        for i in range(self.parts): #z[0]*parts to 600k/parts
            decoded_split = self.onshore_split_decoders[i](split_global_decoded[i])
            decoded_splits.append(decoded_split)

        # Concatenate the decoded parts
        predicted_y = torch.cat(decoded_splits, dim=1) #600k/parts to 600k
        predicted_y = predicted_y[:, :self.xy] #crop to original size
        return predicted_y

        #final correction based on local deformation
        #filter mesh deformation with mask of the onshore grid
        #then below steps:
        # dz_x_y = torch.mul(dz_red,predicted_y)
        # # dy = self.remap(dz_x_y)
        # corrected_y = torch.add(predicted_y,dz_x_y)
        # return corrected_y
        #TODO: come up with some sort of final correction

class BuildTsunamiAE():
    @ex.capture
    def __init__(self,
                MLDir =None, #directory to save model
                reg =None, #regularization
                train_size =None, #training size
                test_size =None, #test size
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
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def configure_scheduler(self):
            self.scheduler = StepLR(self.optimizer, self.step_size, self.gamma)

    def dataloader(self,dataset):
        # tensor_dataset = TensorDataset(torch.flip(torch.Tensor(dataset),dims=[2])) #alternatively use TensorDataset(torch.from_numpy(dataset))
        tensor_dataset = TensorDataset(torch.Tensor(dataset)) #alternatively use TensorDataset(torch.from_numpy(dataset))
        return DataLoader(tensor_dataset, batch_size=self.batch_size, shuffle = False)

    def get_dataloader(self, any_data):
        # # #shuffle any data before splitting
        # np.random.seed(self.seed) #to try a different consistent shuffle across datasets - offshore,onshore,deform
        # np.random.shuffle(any_data) #and see if helps in training
        train_data = any_data[0 : int(len(any_data)*0.99*self.split)]
        val_data = any_data[int(len(any_data)*0.99*self.split) : int(len(any_data)*self.split)]
        test_data = any_data[int(len(any_data)*self.split) : ]

        train_loader = self.dataloader(train_data)
        val_loader = self.dataloader(val_data)
        test_loader = self.dataloader(test_data)

        return train_loader, val_loader, test_loader
       
    def plot_save_loss(self):
            plt.plot(self.train_epoch_losses, color='blue')
            plt.plot(self.val_epoch_losses, color='red')
            plt.plot(self.test_epoch_losses, color='green')
            plt.axvline(x=self.min_epoch, color='black', linestyle='--')
            plt.text(self.min_epoch, .001, self.min_epoch, fontsize=12)
            plt.legend(['train', 'val', 'test'], loc='upper left')
            plt.title(f"Training loss for Nofold")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.yscale('log')
            if self.job == 'couple':
                plt.savefig(f'{self.MLDir}/model/{self.reg}/plot/{self.job}_loss_off{self.channels_off}_on{self.channels_on}_{self.train_size}.png')
                ex.add_artifact(f'{self.MLDir}/model/{self.reg}/plot/{self.job}_loss_off{self.channels_off}_on{self.channels_on}_{self.train_size}.png')

            else:
                plt.savefig(f'{self.MLDir}/model/{self.reg}/plot/{self.job}_loss_ch_{self.channels}_{self.train_size}.png')   
                ex.add_artifact(f'{self.MLDir}/model/{self.reg}/plot/{self.job}_loss_ch_{self.channels}_{self.train_size}.png')
            plt.clf()

    def custom_loss_off(self,recon,true): # recon and true are tensors of same shape as batch,features,seq
        loss = torch.mean(torch.abs(recon-true)**3) + 0.5*torch.mean(torch.abs(recon-true)**2)
        return loss
    
    @ex.capture
    def pretrain(self,
                job,
                data,
                split,
                batch_size,
                lr,
                lr_on,
                lr_deform,
                x_dim,
                y_dim,
                parts = None , #no of splits or parts for onshore
                n  = None , #no of offshore gauges or inundated grids
                t  = None , #no of pts of time (480 time steps)
                z  = None, #latent dim for offshore only
                h = None, #hidden state dim for LSTM
                channels = None, #channels for offshore(1DCNN) or #channels for onshore(fully connected)
                nepochs = 100,
                ):
        #type and data
        self.job = job
        self.data = data
        #hyperparameters
        self.split = split
        self.batch_size = batch_size
        self.nepochs = nepochs

        #model parameters/architecture
        self.parts = parts
        self.z = z
        self.h = h
        self.channels = channels
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n = n
        self.t = t

        #initialize model/settings
        if self.job == 'offshore':
            self.model = Autoencoder_offshore(ninputs=self.n, t_len=self.t, ch_list=self.channels, zdim=self.z, hdim=self.h)
            self.criterion = self.custom_loss_off #custom loss function
            self.lr = lr
        elif self.job == 'onshore' :
            self.model = Autoencoder_onshore(xy=self.n, zlist=self.channels)
            self.criterion = self.custom_loss_off 
            self.lr = lr_on
        elif self.job == 'onshoreparts' :
            self.model = AutoencoderSplitOnshore(parts=self.parts,xy=self.n, zlist=self.channels)
            self.criterion = self.custom_loss_off 
            self.lr = lr_on
            AutoencoderSplitOnshore
        elif self.job == 'deform':
            self.model = Autoencoder_deformation(xy=self.n, df_list=self.channels,zdim=self.z)
            # self.criterion = self.custom_loss_off 
            self.lr = lr_deform
        elif self.job == 'deformfull':
            self.model = Autoencoder_deformationf(x=self.x_dim,y=self.y_dim, df_list=self.channels,zdim=self.z)
            # self.criterion = self.custom_loss_off
            self.lr = lr_deform
        
        self.model.to(self.device)
        self.configure_optimizers()
        self.configure_scheduler()


        #load data
        train_loader, val_loader, test_loader = self.get_dataloader(self.data) 
        self.train_epoch_losses, self.val_epoch_losses, self.test_epoch_losses = [], [], []
        
        print('nepochs:',self.nepochs)
        # Train model
        for epoch in range(self.nepochs):

            train_loss, val_loss, test_loss = 0, 0, 0
            for batch_idx, (batch_data,) in enumerate(train_loader):
                self.optimizer.zero_grad()
                batch_data = batch_data.to(self.device)
                recon_data = self.model(batch_data)
                loss = self.criterion(recon_data,batch_data)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                    
            for batch_idx, (batch_data,) in enumerate(val_loader):
                batch_data = batch_data.to(self.device)
                recon_data = self.model(batch_data)
                vloss = self.criterion(recon_data, batch_data)
                val_loss += vloss.item()
            
            for batch_idx, (batch_data,) in enumerate(test_loader):
                batch_data = batch_data.to(self.device)
                recon_data = self.model(batch_data)
                tloss = self.criterion(recon_data, batch_data)
                test_loss += tloss.item()

            avg_train_ls = train_loss/len(train_loader)
            avg_val_ls = val_loss/len(val_loader)
            avg_test_ls = test_loss/len(test_loader)

            #log to neptune
            log2neptune = True
            if log2neptune:
                run[f'train/{self.job}/epochloss'].append(avg_train_ls)
                run[f'val/{self.job}/epochloss'].append(avg_val_ls)
                run[f'test/{self.job}/epochloss'].append(avg_test_ls)

            if self.verbose:
                print(f'epoch:{epoch},training loss:{avg_train_ls:.5f},val loss:{avg_val_ls:.5f},test loss:{avg_test_ls:.5f}', end='\r')
            
            self.train_epoch_losses.append(avg_train_ls)
            self.val_epoch_losses.append(avg_val_ls)
            self.test_epoch_losses.append(avg_test_ls)
            
            #save model a sepcific intermediate epoch
            # if epoch % 100 == 0 :#and epoch >= 800:
            #     torch.save(self.model, f'{self.MLDir}/model/{self.reg}/out/model_{self.job}_ch_{self.channels}_epoch_{epoch}_{self.train_size}.pt')
            
            #overwrite epochs where val + test loss are the minimum and mark in plot below:
            if epoch == 0:
                min_loss = (val_loss + test_loss)/(len(val_loader)+len(test_loader))
                min_epoch = epoch
            elif (val_loss + test_loss)/(len(val_loader)+len(test_loader)) < min_loss:
                min_loss = (val_loss + test_loss)/(len(val_loader)+len(test_loader))
                min_epoch = epoch
                torch.save(self.model, f'{self.MLDir}/model/{self.reg}/out/model_{self.job}_ch_{self.channels}_minepoch_{self.train_size}.pt')
            
            #at last epoch
            if epoch == self.nepochs-1:
                print(f'epoch:{epoch},training loss:{avg_train_ls:.5f},val loss:{avg_val_ls:.5f},test loss:{avg_test_ls:.5f}', end='\r')
                torch.save(self.model, f'{self.MLDir}/model/{self.reg}/out/model_{self.job}_ch_{self.channels}_epoch_{epoch}_{self.train_size}.pt')

            #early stopping
            if epoch - min_epoch > self.es_gap:
                print('early stopping at epoch:',epoch)
                # torch.save(self.model, f'{self.MLDir}/model/{self.reg}/out/model_{self.job}_ch_{self.channels}_estop_{epoch}_{self.train_size}.pt')
                # ex.log_scalar(f'es_epoch_{self.job}/', epoch)
                break
        
        self.min_epoch = min_epoch
        print('min loss at epoch:',self.min_epoch, 'min loss:',min_loss)
        #save model as artifact
        ex.add_artifact(filename=f'{self.MLDir}/model/{self.reg}/out/model_{self.job}_ch_{self.channels}_minepoch_{self.train_size}.pt')

        #plot and save loss as png and npy
        self.plot_save_loss()
        np.save(f'{self.MLDir}/model/{self.reg}/out/train_loss_{self.job}_ch_{self.channels}_{self.train_size}.npy', self.train_epoch_losses)
        np.save(f'{self.MLDir}/model/{self.reg}/out/test_loss_{self.job}_ch_{self.channels}_{self.train_size}.npy', self.test_epoch_losses)

        #logging
        ex.log_scalar(f'min_loss_{self.job}/', min_loss)
        ex.log_scalar(f'min_epoch_{self.job}/', min_epoch) 

    @ex.capture
    def finetuneAE(self,
            data_in, #training data offshore
            data_deform, #training data deformation
            data_deformfull, #training data deformation
            data_out, #training data offshore
            split, #test data onshore
            batch_size, #batch size onshore
            nepochs,
            lr_couple,
            channels_off = [64,128,256], #channels for offshore(1DCNN)
            channels_on = [64,64], #channels for onshore(fully connected)
            channels_deform = [16,32,64,128], #channels for deformation(1DCNN)
            couple_epochs = [None,None], # epochs for offshore and onshore model coupling otherwise min loss epoch is used
            interface_layers=2,
            tune_nlayers = 1,
            parts = None,
            n = None,
            ):
        super().__init__()
        #data
        self.data_in = data_in
        self.data_deform = data_deform
        self.data_deformfull = data_deformfull
        self.data_out = data_out
        #hyperparameters
        self.split = split
        self.batch_size = batch_size
        self.nepochs = nepochs
        self.lr = lr_couple
        #model parameters/architecture
        self.channels_off = channels_off
        self.channels_on = channels_on
        self.channels_deform = channels_deform
        self.couple_epochs = couple_epochs
        self.interface_layers = interface_layers
        self.tune_nlayers = tune_nlayers
        self.parts = parts
        self.xy = n
        self.job = 'couple'
        self.step_size = 100

        #load model
        if self.couple_epochs[0] == None :
            self.offshore_model = torch.load(f"{self.MLDir}/model/{self.reg}/out/model_offshore_ch_{self.channels_off}_minepoch_{self.train_size}.pt")
            self.deform_model = torch.load(f"{self.MLDir}/model/{self.reg}/out/model_deformfull_ch_{self.channels_deform}_minepoch_{self.train_size}.pt")
            self.onshore_model = torch.load(f"{self.MLDir}/model/{self.reg}/out/model_onshoreparts_ch_{self.channels_on}_minepoch_{self.train_size}.pt")
        elif self.couple_epochs[0] != None :
            self.offshore_model = torch.load(f"{self.MLDir}/model/{self.reg}/out/model_offshore_ch_{self.channels_off}_epoch_{self.couple_epochs[0]}_{self.train_size}.pt")
            self.deform_model = torch.load(f"{self.MLDir}/model/{self.reg}/out/model_deformfull_ch_{self.channels_deform}_epoch_{self.couple_epochs[1]}_{self.train_size}.pt")
            self.onshore_model = torch.load(f"{self.MLDir}/model/{self.reg}/out/model_onshoreparts_ch_{self.channels_on}_epoch_{self.couple_epochs[2]}_{self.train_size}.pt")

        print(self.interface_layers)    
        # Initialize model
        self.model = Autoencoder_coupled2(self.offshore_model,
                                         self.onshore_model, 
                                         self.deform_model,
                                         self.interface_layers,
                                         self.tune_nlayers,
                                         self.parts,
                                         self.xy)
        self.criterion = self.custom_loss_off #custom loss function
        self.model.to(self.device)

        self.configure_optimizers()
        self.configure_scheduler()

        #load data
        train_loader_in, val_loader_in, test_loader_in = self.get_dataloader(self.data_in) 
        train_loader_deform, val_loader_deform, test_loader_deform = self.get_dataloader(self.data_deform)
        train_loader_deformfull, val_loader_deformfull, test_loader_deformfull = self.get_dataloader(self.data_deformfull) 
        train_loader_out, val_loader_out, test_loader_out = self.get_dataloader(self.data_out) 
        self.train_epoch_losses, self.val_epoch_losses, self.test_epoch_losses = [], [], []

        # Train model
        for epoch in range(self.nepochs):
            train_loss, val_loss, test_loss = 0, 0, 0
            for batch_idx,(batch_data_in,batch_data_deform,batch_data_deformfull,batch_data_out) in enumerate(zip(train_loader_in,train_loader_deform,train_loader_deformfull,train_loader_out)):
                self.optimizer.zero_grad()
                batch_data_in = batch_data_in[0].to(self.device)
                batch_data_deform = batch_data_deform[0].to(self.device)
                batch_data_deformfull = batch_data_deformfull[0].to(self.device)
                batch_data_out = batch_data_out[0].to(self.device)
                recon_data = self.model(batch_data_in,batch_data_deform,batch_data_deformfull)
                loss = self.criterion(recon_data, batch_data_out)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                    
            for batch_idx,(batch_data_in,batch_data_deform,batch_data_deformfull,batch_data_out)  in enumerate(zip(val_loader_in,val_loader_deform,val_loader_deformfull,val_loader_out)):
                batch_data_in = batch_data_in[0].to(self.device)
                batch_data_deform = batch_data_deform[0].to(self.device)
                batch_data_deformfull = batch_data_deformfull[0].to(self.device)
                batch_data_out = batch_data_out[0].to(self.device)
                recon_data = self.model(batch_data_in,batch_data_deform,batch_data_deformfull)
                vloss = self.criterion(recon_data, batch_data_out)
                val_loss += vloss.item()

            for batch_idx,(batch_data_in,batch_data_deform,batch_data_deformfull,batch_data_out)  in enumerate(zip(test_loader_in,test_loader_deform,test_loader_deformfull,test_loader_out)):
                batch_data_in = batch_data_in[0].to(self.device)
                batch_data_deform = batch_data_deform[0].to(self.device)
                batch_data_deformfull = batch_data_deformfull[0].to(self.device)
                batch_data_out = batch_data_out[0].to(self.device)
                recon_data = self.model(batch_data_in,batch_data_deform,batch_data_deformfull)
                tloss = self.criterion(recon_data, batch_data_out)
                test_loss += tloss.item()

            avg_train_ls = train_loss/len(train_loader_in)
            avg_val_ls = val_loss/len(val_loader_in)
            avg_test_ls = test_loss/len(test_loader_in)

            #log to neptune
            log2neptune = True
            if log2neptune:
                run[f'train/{self.job}/epochloss'].append(avg_train_ls)
                run[f'val/{self.job}/epochloss'].append(avg_val_ls)
                run[f'test/{self.job}/epochloss'].append(avg_test_ls)

            if self.verbose:
                print(f'epoch:{epoch},training loss:{avg_train_ls:.5f},val loss:{avg_val_ls:.5f},test loss:{avg_test_ls:.5f}', end='\r')
            
            self.train_epoch_losses.append(avg_train_ls)
            self.val_epoch_losses.append(avg_val_ls)
            self.test_epoch_losses.append(avg_test_ls)
            
            #save model a sepcific intermediate epoch
            # if epoch % 100 == 0 :#and epoch >= 800:
            #     torch.save(self.model, f'{self.MLDir}/model/{self.reg}/out/model_{self.job}_off{self.channels_off}_on{self.channels_on}_epoch_{epoch}_{self.train_size}.pt')
        
            #overwrite epochs where val + test loss are the minimum and mark in plot below:
            if epoch == 0:
                min_loss = avg_val_ls + avg_test_ls
                min_epoch = epoch
            elif avg_val_ls + avg_test_ls < min_loss:
                min_loss = avg_val_ls + avg_test_ls
                min_epoch = epoch
                torch.save(self.model, f'{self.MLDir}/model/{self.reg}/out/model_{self.job}_off{self.channels_off}_on{self.channels_on}_minepoch_{self.train_size}.pt')
            
            #at last epoch
            if epoch == self.nepochs-1:
                print(f'epoch:{epoch},training loss:{avg_train_ls:.5f},val loss:{avg_val_ls:.5f},test loss:{avg_test_ls:.5f}', end='\r')
                torch.save(self.model, f'{self.MLDir}/model/{self.reg}/out/model_{self.job}_off{self.channels_off}_on{self.channels_on}_epoch_{epoch}_{self.train_size}.pt')

            #early stopping
            if epoch - min_epoch > self.es_gap:
                print('early stopping at epoch:',epoch, 'min loss:',min_loss)
                # torch.save(self.model, f'{self.MLDir}/model/{self.reg}/out/model_{self.job}_off{self.channels_off}_on{self.channels_on}_estop_{epoch}_{self.train_size}.pt')
                # ex.log_scalar(f'es_epoch_{self.job}/', min_epoch)
                break
        
        self.min_epoch = min_epoch
        print('min loss at epoch:',self.min_epoch)
        #save model as artifact
        ex.add_artifact(filename=f'{self.MLDir}/model/{self.reg}/out/model_{self.job}_off{self.channels_off}_on{self.channels_on}_minepoch_{self.train_size}.pt')

        #plot and save loss as png and npy
        self.plot_save_loss()
        np.save(f'{self.MLDir}/model/{self.reg}/out/train_loss_{self.job}_ch_{self.channels}_{self.train_size}.npy', self.train_epoch_losses)
        np.save(f'{self.MLDir}/model/{self.reg}/out/test_loss_{self.job}_ch_{self.channels}_{self.train_size}.npy', self.test_epoch_losses)

        #logging
        ex.log_scalar(f'min_loss_{self.job}/', min_loss)
        ex.log_scalar(f'min_epoch_{self.job}/', min_epoch) 


    @ex.capture   
    def evaluateAE(self,
                    data_in, #training data offshore
                    data_deform, #training data deformation
                    data_out, #training data onshore
                    model_def, #model feature inputs and outputs
                    channels_off = [64,128,256], #channels for offshore(1DCNN)
                    channels_on = [64,64], #channels for onshore(fully connected)
                    epoch =  None,#selected epoch
                    batch_size_on = 100, #depends on GPU memory
                    control_points = [], #control points for evaluation
                    threshold = 0.1, #threshold for evaluation
                    device =  torch.device("cpu"),
                    ):
        
        self.job = 'evaluate'
        self.batch_size = batch_size_on
        self.channels_off = channels_off
        self.channels_on = channels_on
        self.device = device
             
        #read model from file for testing
        if epoch is None:
            model = torch.load(f'{self.MLDir}/model/{self.reg}/out/model_couple_off{self.channels_off}_on{self.channels_on}_minepoch_{self.train_size}.pt',map_location=torch.device('cpu'))
        else:
            model = torch.load(f'{self.MLDir}/model/{self.reg}/out/model_couple_off{self.channels_off}_on{self.channels_on}_epoch_{epoch}_{self.train_size}.pt',map_location=torch.device('cpu')) 
        model.eval()

        # print('model summary.....')
        # summary(model,[(model_def[0],model_def[1]),(model_def[2])])

        #load data
        # data_in = torch.tensor(data_in, dtype=torch.float32).to('cpu')
        # data_deform = torch.tensor(data_deform, dtype=torch.float32).to('cpu')
        # data_out = torch.tensor(data_out, dtype=torch.float32).to('cpu')
        # with torch.no_grad():
        #     recon_data= model(data_in,data_deform)
        #     loss = self.criterion(recon_data, data_out)
        # print(f"test loss: {loss / len(data_in):.5f}")
        # predic = recon_data.cpu().numpy()

        test_loader_in = self.dataloader(data_in)
        test_loader_deform = self.dataloader(data_deform)
        test_loader_out = self.dataloader(data_out)
        predic = np.zeros(data_out.shape)

        # Test model
        with torch.no_grad():
            test_loss = 0
            for batch_idx,(batch_data_in,batch_data_deform,batch_data_out) in enumerate(zip(test_loader_in,test_loader_deform,test_loader_out)):
                batch_data_in = batch_data_in[0].to(self.device)
                batch_data_deform = batch_data_deform[0].to(self.device)
                batch_data_out = batch_data_out[0].to(self.device)

                recon_data = model(batch_data_in,batch_data_deform)
                loss = self.criterion(recon_data, batch_data_out)
                test_loss += loss.item()
                predic[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size] = recon_data.cpu().numpy()
            print(f"test loss: {test_loss / len(test_loader_in):.5f}")

        # Plot results max height for all events
        test_max = np.max(data_out,axis=(1))
        recon_max = np.max(predic,axis=(1))
        r2maxdepth = r2_score(test_max, recon_max)
        
        #plot max depth for all events
        plt.figure(figsize=(5, 5))
        plt.scatter(test_max, recon_max, s=1)
        plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, color='red')
        plt.title(f"Max height for each events")
        plt.text(10,5,f"R Squared: {r2maxdepth:.5f} ", fontsize=12)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.grid()
        plt.xlabel('True')
        plt.ylabel('Reconstructed')
        plt.savefig(f'{self.MLDir}/model/{self.reg}/plot/model_coupled_off{self.channels_off}_on{self.channels_on}_{self.train_size}_maxdepth_testsize{self.test_size}.png')
        ex.add_artifact(filename=f'{self.MLDir}/model/{self.reg}/plot/model_coupled_off{self.channels_off}_on{self.channels_on}_{self.train_size}_maxdepth_testsize{self.test_size}.png')
        #first calculate location index of control points for given lat and lon
        locindices = get_idx_from_latlon(control_points)

        #evaluation table
        eve_perf = []
        true_list = []
        pred_list = []
        er_list = []

        #mse_val,r2_val,pt_er,KCap,Ksmall,truecount,predcount
        test_ids = np.loadtxt(f'{self.MLDir}/data/events/shuffled_events_test_{self.reg}_{self.test_size}.txt',dtype='str')
        for eve_no,eve in enumerate(test_ids):
            scores = calc_scores(data_out[eve_no,:], predic[eve_no,:],locindices,threshold)
            eve_perf.append([scores[0],scores[1],scores[5],#scores[4], #mse,r2,GFIT
                            np.count_nonzero(data_out[eve_no,:]), #true count
                            np.count_nonzero(predic[eve_no,:])]) #pred count
            true_list.append(scores[2])
            pred_list.append(scores[3])
            er_list.append(scores[4])

        #count of events less than 
        eve_perf = np.array(eve_perf)
        true_list = np.array(true_list)
        pred_list = np.array(pred_list)
        er_list = np.array(er_list)

        #combine columns of true,pred,er into 12 column array
        true_pred_er = np.column_stack((true_list,pred_list,er_list))
        np.savetxt(f'{self.MLDir}/model/{self.reg}/out/model_coupled_off{self.channels_off}_on{self.channels_on}_{self.train_size}_true_pred_er_testsize{self.test_size}.txt',true_pred_er,fmt='%1.4f')
        ex.add_artifact(filename=f'{self.MLDir}/model/{self.reg}/out/model_coupled_off{self.channels_off}_on{self.channels_on}_{self.train_size}_true_pred_er_testsize{self.test_size}.txt')
        
        #print overall performance metrics for whole training exercise and evaluation work : mseoverall, K, k, r2maxdepth
        # Calculate mseoverall and r2maxdepth
        mseoverall = mean_squared_error(data_out, predic)
        gfitoverall = np.mean(Gfit(data_out, predic))
        print(f"mseoverall: {mseoverall:.4f}")
        print(f"r2maxdepth: {r2maxdepth:.4f}")
        print(f"gfitoverallmean: {gfitoverall:.4f}")
        #TODO: add per event evaluation for discovery and analysis

        
        #log metrics to sacred
        ex.log_scalar('mseoverall',mseoverall)
        ex.log_scalar('r2maxdepth',r2maxdepth)
        ex.log_scalar('gfitoverall',gfitoverall)
      
        #plot error at each location
        print('plotting error at each control points')
        plt.figure(figsize=(15, 30))
        #add to main plot the mse and r2 to the plot at the top
        plt.suptitle(f"mseoverall: {mseoverall:.5f},r2maxdepth: {r2maxdepth:.5f}gfitoverall:{gfitoverall:.4f},testsize: {self.test_size}",fontsize=25)
        
        
        #error charts
        for i in range(len(locindices)):
            plt.subplot(6,2,i+1)
            plt.hist(er_list[er_list[:,i]!=0,i],bins=50)
            #set x axis to be the same for all subplots
            plt.xlim(-2,2)
            #calculate hit and mis for each location based on depth of true and prediction
            #events crossing the threshold of 0.2 are considered flooded
            neve = np.count_nonzero(true_pred_er[:,i]>threshold) # no of flooded grid points in the event
            neve_pred = np.count_nonzero(true_pred_er[:,i+4]>threshold) # no of flooded grid points in the prediction
            #true positive: true>0.2 and pred>0.2 if threshold is 0.2
            if neve == 0:
                TP = -999
                FN = -999
            else:
            #true positive: true>0.2 and pred>0.2
                TP = np.count_nonzero((true_pred_er[:,i]>threshold) & (true_pred_er[:,i+4]>threshold))/(neve)
                FN = np.count_nonzero((true_pred_er[:,i]>threshold) & (true_pred_er[:,i+4]<=threshold))/(neve)
            TN = np.count_nonzero((true_pred_er[:,i]<=threshold) & (true_pred_er[:,i+4]<=threshold))/(len(true_pred_er[:,i])-neve)
            FP = np.count_nonzero((true_pred_er[:,i]<=threshold) & (true_pred_er[:,i+4]>threshold))/(len(true_pred_er[:,i])-neve)
            plt.title(f"Control Location:{i+1},No of flood events:T{neve}/P{neve_pred}/len:{len(true_pred_er[:,i])}")
            plt.text(0.78, 0.9, f" TP: {TP:.2f}, TN: {TN:.2f}", horizontalalignment='center',verticalalignment='center', transform=plt.gca().transAxes,fontsize=12)
            plt.text(0.78, 0.75, f"FP: {FP:.2f}, FN: {FN:.2f}", horizontalalignment='center',verticalalignment='center', transform=plt.gca().transAxes,fontsize=12)
            plt.xlabel('Error')
            plt.ylabel('Count')
        plt.savefig(f'{self.MLDir}/model/{self.reg}/plot/model_coupled_off{channels_off}_on{channels_on}_{self.train_size}_error_testsize{self.test_size}.png')
        ex.add_artifact(f'{self.MLDir}/model/{self.reg}/plot/model_coupled_off{channels_off}_on{channels_on}_{self.train_size}_error_testsize{self.test_size}.png')

        #plot rel error at each location
        #set true value less than threshold to zero
        print('plotting rel error at each control points')
        true_list[true_list<threshold] = 0
        #calculate rel error for each row/column variable where er is not zero
        reler_list = np.where(true_list != 0, er_list / true_list, 0.0)

        plt.figure(figsize=(15, 30))
        #add to main plot the mse and r2 to the plot at the top
        plt.suptitle(f"mseoverall: {mseoverall:.5f},r2maxdepth: {r2maxdepth:.5f}gfitoverall:{gfitoverall:.4f},testsize: {self.test_size}",fontsize=25)
        
        
        #error charts
        for i in range(len(locindices)):
            plt.subplot(6,2,i+1)

            plt.hist(reler_list[reler_list[:,i]!=0,i],bins=50)
            #set x axis to be the same for all subplots
            plt.xlim(-2,2)
            #calculate hit and mis for each location based on depth of true and prediction
            #events crossing the threshold of 0.2 are considered flooded
            neve = np.count_nonzero(true_pred_er[:,i]>threshold) # no of flooded grid points in the event
            neve_pred = np.count_nonzero(true_pred_er[:,i+4]>threshold) # no of flooded grid points in the prediction
            #true positive: true>0.2 and pred>0.2 if threshold is 0.2
            if neve == 0:
                TP = -999
                FN = -999
            else:
            #true positive: true>0.2 and pred>0.2
                TP = np.count_nonzero((true_pred_er[:,i]>threshold) & (true_pred_er[:,i+4]>threshold))/(neve)
                FN = np.count_nonzero((true_pred_er[:,i]>threshold) & (true_pred_er[:,i+4]<=threshold))/(neve)
            TN = np.count_nonzero((true_pred_er[:,i]<=threshold) & (true_pred_er[:,i+4]<=threshold))/(len(true_pred_er[:,i])-neve)
            FP = np.count_nonzero((true_pred_er[:,i]<=threshold) & (true_pred_er[:,i+4]>threshold))/(len(true_pred_er[:,i])-neve)
            plt.title(f"Control Location:{i+1},No of flood events:T{neve}/P{neve_pred}/len:{len(true_pred_er[:,i])}")
            plt.text(0.78, 0.9, f" TP: {TP:.2f}, TN: {TN:.2f}", horizontalalignment='center',verticalalignment='center', transform=plt.gca().transAxes,fontsize=12)
            plt.text(0.78, 0.75, f"FP: {FP:.2f}, FN: {FN:.2f}", horizontalalignment='center',verticalalignment='center', transform=plt.gca().transAxes,fontsize=12)
            plt.xlabel('Error')
            plt.ylabel('Count')
        plt.savefig(f'{self.MLDir}/model/{self.reg}/plot/model_coupled_off{channels_off}_on{channels_on}_{self.train_size}_relerror_testsize{self.test_size}.png')
        ex.add_artifact(f'{self.MLDir}/model/{self.reg}/plot/model_coupled_off{channels_off}_on{channels_on}_{self.train_size}_relerror_testsize{self.test_size}.png')

def calc_scores(true,pred,locindices,threshold): #for each event
    #only test where there is significant flooding
    true[true<threshold] = 0
    pred[pred<threshold] = 0
    mse_val = mean_squared_error(true,pred)
    r2_val = r2_score(true,pred)
    Gfit_val = Gfit_one(true,pred)
    pt_er = true[locindices] - pred[locindices]

    #TODO: Aidan's no K and k small not working check again
    # sel = true>=threshold
    # true = true[sel]
    # pred = pred[sel]
    # #to avoid division by zero    
    # pred[pred==0]=1e-10
    # #calculate Aidan's no K and k small 
    # ratio = true/pred
    # logval = np.log(ratio)
    # logval_sq = logval**2

    # logk = logval.mean()

    # logksmall = ((logval_sq.mean()) - (logk**2))**0.5
    # KCap = np.exp(logk)
    # Ksmall = np.exp(logksmall)

    return mse_val,r2_val,true[locindices],pred[locindices],pt_er,Gfit_val 

def Gfit(obs, pred): #a normalized least-squares per event in first dimensions
    # print('obs shape', obs.shape,len(obs))
    Gtable = np.zeros(len(obs))
    for i in range(len(obs)):      
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
    print(indices)
    return indices

def process_ts(file):
    #read data
    data = xr.open_dataset(file)
    ts = data['eta'].values
    maxTS = ts.max(axis=0)
    minTS = ts.min(axis=0)
    gperiod = []
    gpolarity = []
    greturn_code = []

    for g in range(87):
        #find peaks(positive and negative)
        ppeaks, _ = scipy.signal.find_peaks(ts[:,g], height=0.05,distance=50)
        npeaks, _ = scipy.signal.find_peaks(-ts[:,g], height=0.05,distance=50)

        #find polarity of wave based on positive and negative peaks indices
        if len(ppeaks)==0 and len(npeaks)==0:
            polarity = '0'
        elif len(ppeaks)==0:
            polarity = '-1'
        elif len(npeaks)==0:
            polarity = '+1'
        elif ppeaks[0]<npeaks[0]:
            polarity = '+1'
        elif ppeaks[0]>npeaks[0]:
            polarity = '-1'
        else:
            polarity = '0'

        #find waveperiod
        if polarity == '0':
            waveperiod = 0
        elif polarity == '+1':
            if len(ppeaks)==1:
                waveperiod = 0
            else:
                waveperiod = (ppeaks[1]-ppeaks[0])*30
        elif polarity == '-1':
            if len(npeaks)==1:
                waveperiod = 0
            else:
                waveperiod = (npeaks[1]-npeaks[0])*30

        #return code
        if polarity == '0':
            return_code = 1
        elif polarity == '+1' or polarity == '-1':
            return_code = 3

        gperiod.append(waveperiod)
        gpolarity.append(polarity)
        greturn_code.append(return_code)

    gperiod = np.array(gperiod)
    gpolarity = np.array(gpolarity)
    greturn_code = np.array(greturn_code)


    #compile to dataframe #ID lon lat depth max_ssh min_ssh period polarity return_code
    df = pd.DataFrame({'ID':np.arange(87)+1,
                        'lon':data['longitude'].values,
                        'lat':data['latitude'].values,
                        'depth':data['deformed_bathy'].values,
                        'max_ssh':maxTS,
                        'min_ssh':minTS,
                        'period':gperiod,
                        'polarity':gpolarity,
                        'return_code':greturn_code})

    #save to csv
    df.to_csv(file.replace('grid0_ts.nc','grid0_ts.nc.offshore.txt'),index=False,sep='\t')

def sample_train_events(data, #input dataframe with event info 
                        importance_column='mean_prob', #column name for importance weighted sampling
                        samples_per_bin=10, #no of events to sample per bin
                        bin_def = [0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0], #bin width for wave height 
                        ): 

    #define bin edges
    bin_start = bin_def[:-1]
    bin_end = bin_def[1:]

    if np.any((data['mean_prob'] < 0) | (data[importance_column] < 0)):
        raise ValueError('event_rates and importance parameter must be nonnegative')
    
    sample = []
    for bin in list(zip(bin_start, bin_end)):
        #get events in this bin
        events_in_bin = data[(data['max_off'] >= bin[0]) & (data['max_off'] < bin[1])]
   
        if len(events_in_bin) <= samples_per_bin:
            print('Less scenario to sample in this bin',bin,' -- need to be careful using samples')
        
        rate_with_this_bin = np.sum(events_in_bin['mean_prob'] )    
        events_in_bin_copy = events_in_bin.copy()

        #sort by lat and lon
        events_in_bin_copy.sort_values(by=['lat', 'lon'], inplace=True)

        if importance_column == 'gridcount' or importance_column == 'LocationCount':
            # Print count of events per grid and print grid ID with the minimum count
            grid_counts = events_in_bin_copy.groupby('grid_id').count()['id']

            # Add the weights column
            events_in_bin_copy['norm_wt'] = events_in_bin_copy['grid_id'].map(lambda x: 1 / grid_counts[x])
            events_in_bin_copy['norm_wt'] /= np.sum(events_in_bin_copy['norm_wt'])

            if samples_per_bin > 0 and np.sum(events_in_bin_copy['norm_wt']) > 0:
                #inverse wted sampling grid count
                sampled_ids = np.random.choice(events_in_bin['id'],
                                                size=samples_per_bin,
                                                p=events_in_bin_copy['norm_wt'],
                                                replace=False,
                                                )
       
        else:
            events_in_bin_copy.loc[:, 'norm_wt'] = events_in_bin[importance_column] / np.sum(events_in_bin[importance_column])           
            if samples_per_bin > 0 and np.sum(events_in_bin_copy['norm_wt']) > 0:
                #sample events from this bin weighted by importance
                sampled_ids = np.random.choice(events_in_bin['id'],
                                                size=samples_per_bin,
                                                p=events_in_bin_copy['norm_wt'],
                                                replace=False,
                                                )
                       
        #get the sampled events
        sample.append(pd.DataFrame({
            'id': sampled_ids,
            'bin_start': np.repeat(bin[0], samples_per_bin),
            'bin_end': np.repeat(bin[1], samples_per_bin),
            'rate_with_this_bin': np.repeat(rate_with_this_bin, samples_per_bin)
        }))
    
    return sample

@ex.capture
def sample_events(wt_para = 'gridcount', #'LocationCount', 'mean_prob', 'importance', 'uniform_wt', 'gridcount'
                  samples_per_bin = 15,
                  bin_splits = 12,
                  reg = 'SR',
                  reg_gaugeno = '54',
                  MLDir = None,
                  ):
    
    data = pd.read_csv(MLDir + f'data/info/sampling_input_{reg}_{reg_gaugeno}.csv')

    'split sampling in two steps for event type 0 and 1 then merge'
    sample_step0 = sample_train_events(data.groupby('event_type').get_group(0),
                                    importance_column=wt_para,
                                    samples_per_bin=samples_per_bin,
                                    bin_def = np.append(np.linspace(0,3,bin_splits), 99))

    sample_step1 = sample_train_events(data.groupby('event_type').get_group(1),
                                    importance_column=wt_para, 
                                    samples_per_bin=samples_per_bin*4,
                                    bin_def = np.append(np.linspace(0,3,bin_splits), 99))

    sample_test = pd.concat([pd.concat(sample_step0, axis=0), pd.concat(sample_step1, axis=0)], axis=0)

    #check unique events in sample
    sample_len = len(sample_test['id'].unique())
    print(sample_len,'out of ',len(sample_test))

    #merge columns from combined to sample_test based on id
    sample_test = pd.merge(sample_test, data, on='id', how='left')

    #plot same but in multiple subplots as row
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].hist(sample_test[sample_test['event_type']==0]['max_off'], bins=bin_splits, alpha=0.5, label='near - max off')
    axs[0].hist(sample_test[sample_test['event_type']==0]['max'], bins=bin_splits, alpha=0.5, label='near - max inun')
    axs[0].text(0.5, 0.5, 'count of type 0 events: '+str(len(sample_test[sample_test['event_type']==0])))
    axs[0].legend(loc='upper right')

    axs[1].hist(sample_test[sample_test['event_type']==1]['max_off'], bins=bin_splits, alpha=0.5, label='far - max off')
    axs[1].hist(sample_test[sample_test['event_type']==1]['max'], bins=bin_splits, alpha=0.5, label='far - max inun')
    axs[1].text(0.5, 0.5,'count of type 0 events: '+str(len(sample_test[sample_test['event_type']==1])))
    axs[1].legend(loc='upper right')

    axs[2].hist(sample_test['max_off'], bins=bin_splits, alpha=0.5, label='max off')
    axs[2].hist(sample_test['max'], bins=bin_splits, alpha=0.5, label='max inun')
    axs[2].text(0.5, 0.5, 'count of events: '+str(len(sample_test)))
    axs[2].legend(loc='upper right')
    plt.savefig(MLDir + f'/model/{reg}/plot/sampledist_events{str(sample_len)}_{reg}_{reg_gaugeno}.png', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(15,10))
    ax.title.set_text('Sampled events by: ' + wt_para)
    ax = plt.scatter(sample_test[sample_test['event_type']==1]['lon'], sample_test[sample_test['event_type']==1]['lat'], alpha=0.75, label='far',s=4)
    ax = plt.scatter(sample_test[sample_test['event_type']==0]['lon'], sample_test[sample_test['event_type']==0]['lat'], alpha=0.75, label='near', s=5)
    plt.legend(loc='upper right')
    plt.xlim(10, 35)
    plt.ylim(30, 43)
    ax = plt.gca()
    cx.add_basemap(ax,crs='EPSG:4326', source=cx.providers.Stamen.TonerLite)
    plt.savefig(MLDir + f'/model/{reg}/plot/samplemap_events{str(sample_len)}_{reg}_{reg_gaugeno}.png', bbox_inches='tight')

    #save list of ids to txt file
    path = MLDir + f'/data/events/sample_events{str(sample_len)}_{reg}_{reg_gaugeno}.txt'
    sample_test['id'].unique().to_csv(path, header=False, index=None)
    
    return sample_len, path

@ex.capture
def read_memmap(MLDir,
                reg,
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
                twindow = None,
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

        # Plot a sample of ts for all gauges for the first event
        plt.figure(figsize=(15, 5))
        plt.plot(t_array_temp[0].T)  # Plot t_array_temp instead of t_array
        plt.title(f"Time series for all gauges for the first event")
        plt.savefig(f'{MLDir}/model/{reg}/plot/ts_{reg}_{size}_firstevent.png')
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
        return t_array_temp, red_d_array, red_dZ_array, dZ_array
    else:
        return t_array, red_d_array, red_dZ_array, dZ_array


