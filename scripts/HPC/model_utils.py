import numpy as np
import pandas as pd
import scipy.signal

import xarray as xr
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import matplotlib
import matplotlib.pyplot as plt
import contextily as cx

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


try:
    MLDir = os.getenv('MLDir')
    SimDir = os.getenv('SimDir')
    reg = sys.argv[1] #CT or SR
    size = sys.argv[2] #eventset size for training
    mode = sys.argv[3] #train or test
    mask_size = sys.argv[4] #eventset size for testing
except:
    raise Exception("*** Must first set environment variable")

#set seed and torch settings
np.random.seed(0)
torch.random.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#build the offshore model
class Autoencoder_offshore(nn.Module):
    def __init__(self,
                 ninputs=5, #number of input channels or gauges
                 t_len = 480, #number of time steps
                 ch_list = [32,64,96], #number of channels in each layer
                 zdim = 50):#number of latent variables
        
        super(Autoencoder_offshore, self).__init__()
        # more channels mean more fine details, more resolution 
        # less channel and layers less likely to overfit so better maximas and minimas
        # more accuracy but slower and more memory and data needed to train
        self.ch_list = ch_list

        # define encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(ninputs, ch_list[0], kernel_size=3, padding=1),   
            nn.LeakyReLU(negative_slope=0.5,inplace=True),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(ch_list[0], ch_list[1], kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.5,inplace=True),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(ch_list[1], ch_list[2], kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.5,inplace=True),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            nn.Flatten(),
            nn.Linear(int(t_len*self.ch_list[-1]/(2**len(ch_list))), zdim),          
        )

        # define decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(zdim,int(t_len*self.ch_list[-1]/(2**len(ch_list)))),
            nn.Unflatten(1, (ch_list[-1], int(t_len/(2**len(ch_list))))) ,
            nn.ConvTranspose1d(ch_list[2], ch_list[1], kernel_size=4, padding=1, stride= 2),  
            nn.LeakyReLU(negative_slope=0.5,inplace=True) ,
            nn.ConvTranspose1d(ch_list[1], ch_list[0], kernel_size=4, padding=1, stride= 2), 
            nn.LeakyReLU(negative_slope=0.5,inplace=True) ,
            nn.ConvTranspose1d(ch_list[0], ninputs, kernel_size=4, padding=1, stride= 2), 
            nn.LeakyReLU(negative_slope=0.5,inplace=True) ,
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

class Autoencoder_coupled(nn.Module):
    def __init__(self,
                offshore_model,
                onshore_model,
                deform_model,
                interface_layers,
                tune_layer,
                **kwargs):
        super(Autoencoder_coupled, self).__init__()

        # Pretrained offshore 
        self.offshore_encoder = offshore_model.encoder
        for i, layer in enumerate(self.offshore_encoder):
            if i < len(self.offshore_encoder) - tune_layer: #all layers except last
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters(): #last layer
                    param.requires_grad = True

        # Pretrained deform 
        self.deform_encoder = deform_model.encoder
        for i, layer in enumerate(self.deform_encoder):
            if i < len(self.deform_encoder) - tune_layer: #all layers except last
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters(): #last layer
                    param.requires_grad = True

        # Interface
        if interface_layers == 1:
            self.connect = nn.Sequential(
                                nn.Linear(
                                    in_features=64, out_features=64
                                ),
                                nn.ReLU(),
            ) 
        elif interface_layers == 2:    
            self.connect = nn.Sequential(
                                    nn.Linear(
                                        in_features=64, out_features=64
                                    ),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Linear(
                                        in_features=64, out_features=64
                                    ),
                                    nn.LeakyReLU(inplace=True),
            )
        # Pretrained onshore model
        self.onshore_decoder = onshore_model.decoder 
        for i, layer in enumerate(self.onshore_decoder):
            if i < tune_layer:
                for param in layer.parameters(): #first layer
                    param.requires_grad = True
            else:
                for param in layer.parameters(): #all layers except first
                    param.requires_grad = False

    def forward(self, x, dz):
        x = self.offshore_encoder(x)
        # dz = self.deform_encoder(dz)
        # x = torch.cat((x, dz), dim=1)
        x = self.connect(x)
        x = self.onshore_decoder(x)
        return x
  
def pretrainAE(job, #offshore  or onshore or deform
            data, #training data 
            test_data, #test data 
            batch_size = 50,
            nepochs = 1000,
            lr = 0.0005,
            n = None, #no of offshore gauges or inundated grids
            t = None, #no of pts of time (480 time steps)
            z = None, #latent dim for offshore only
            channels = None, #channels for offshore(1DCNN) or #channels for onshore(fully connected)
            verbose = False):
    
    train_size = size
    # Create PyTorch DataLoader objects
    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(data[0:int(len(data)*0.99)]))   
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = torch.utils.data.TensorDataset(torch.Tensor(data[int(len(data)*0.99):]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_data))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    if job == 'offshore':
        print('Training Autoencoder:',job,'with channels:',channels, 'latent_dim:', z)
        model = Autoencoder_offshore(ninputs = n ,t_len = t, ch_list = channels, zdim = z)
    elif job == 'onshore' or job == 'deform':
        print('Training Autoencoder:',job,'with channels:',channels)
        model = Autoencoder_onshore(xy = n , zlist = channels)
    
    model.to('cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_epoch_losses = []
    val_epoch_losses = []
    test_epoch_losses = []

    # Train model
    for epoch in range(nepochs):
        train_loss = 0
        val_loss = 0
        test_loss = 0
        for batch_idx, (batch_data,) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_data = batch_data.to('cuda')
            recon_data = model(batch_data)
            loss = criterion(recon_data, batch_data)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
                
        for batch_idx, (batch_data,) in enumerate(val_loader):
            batch_data = batch_data.to('cuda')
            recon_data = model(batch_data)
            vloss = criterion(recon_data, batch_data)
            val_loss += vloss.item()
        
        for batch_idx, (batch_data,) in enumerate(test_loader):
            batch_data = batch_data.to('cuda')
            recon_data = model(batch_data)
            tloss = criterion(recon_data, batch_data)
            test_loss += tloss.item()

        avg_train_ls = train_loss/len(train_loader)
        avg_val_ls = val_loss/len(val_loader)
        avg_test_ls = test_loss/len(test_loader)

        if verbose:
            print(f'epoch:{epoch},training loss:{avg_train_ls:.5f},val loss:{avg_val_ls:.5f},test loss:{avg_test_ls:.5f}', end='\r')
          
        train_epoch_losses.append(avg_train_ls)
        val_epoch_losses.append(avg_val_ls)
        test_epoch_losses.append(avg_test_ls)
        
        #save model a sepcific intermediate epoch
        if epoch % 100 == 0 :#and epoch >= 800:
            torch.save(model, f'{MLDir}/model/{reg}/out/model_{job}_ch_{channels}_epoch_{epoch}_{train_size}.pt')
        
        #overwrite epochs where val + test loss are the minimum and mark in plot below:
        if epoch == 0:
            min_loss = avg_val_ls + avg_test_ls
            min_epoch = epoch
        elif avg_val_ls + avg_test_ls < min_loss:
            min_loss = avg_val_ls + avg_test_ls
            min_epoch = epoch
            torch.save(model, f'{MLDir}/model/{reg}/out/model_{job}_ch_{channels}_minepoch_{train_size}.pt')
        
        #at last epoch
        if epoch == nepochs-1:
            print(f'epoch:{epoch},training loss:{avg_train_ls:.5f},val loss:{avg_val_ls:.5f},test loss:{avg_test_ls:.5f}', end='\r')
            torch.save(model, f'{MLDir}/model/{reg}/out/model_{job}_ch_{channels}_epoch_{epoch}_{train_size}.pt')

    print('min loss at epoch:',min_epoch)
   
    #plot training/val loss #TODO: save as numpy array for later exploration
    plt.plot(train_epoch_losses, color='blue')
    plt.plot(val_epoch_losses, color='red')
    plt.plot(test_epoch_losses, color='green')
    plt.axvline(x=min_epoch, color='black', linestyle='--')
    plt.text(min_epoch, 0.1, min_epoch, fontsize=12)
    plt.legend(['train', 'val', 'test'], loc='upper left')
    plt.title(f"Training loss for Nofold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.savefig(f'{MLDir}/model/{reg}/plot/{job}_loss_ch_{channels}_{train_size}.png')   
    plt.clf()

def finetuneAE(data_in, #training data offshore #TODO: data loader can be a separate module might be more efficient than preprocessing,especially for training and final 53k event eval
            data_deform, #training data deformation
            data_out, #training data offshore
            test_data_in, #test data onshore
            test_data_deform, #test data deformation
            test_data_out, #test data onshore
            batch_size = 50,
            nepochs = 1000,
            lr = 0.00005,
            channels_off = [64,128,256], #channels for offshore(1DCNN)
            channels_on = [64,64], #channels for onshore(fully connected)
            couple_epochs = [None,None], # epochs for offshore and onshore model coupling otherwise min loss epoch is used
            interface_layers=2,
            tune_nlayers = 1,
            verbose = False):
    train_size = size
    # Create PyTorch DataLoader objects
    #input-offshore, 
    train_dataset_in = torch.utils.data.TensorDataset(torch.Tensor(data_in[0:int(len(data_in)*0.99)]))   
    train_loader_in = torch.utils.data.DataLoader(train_dataset_in, batch_size=batch_size, shuffle=False)
    val_dataset_in = torch.utils.data.TensorDataset(torch.Tensor(data_in[int(len(data_in)*0.99):]))
    val_loader_in = torch.utils.data.DataLoader(val_dataset_in, batch_size=batch_size, shuffle=False)
    test_dataset_in = torch.utils.data.TensorDataset(torch.Tensor(test_data_in))
    test_loader_in = torch.utils.data.DataLoader(test_dataset_in, batch_size=batch_size, shuffle=False)
    #input-deformation
    train_dataset_deform = torch.utils.data.TensorDataset(torch.Tensor(data_deform[0:int(len(data_deform)*0.99)]))
    train_loader_deform = torch.utils.data.DataLoader(train_dataset_deform, batch_size=batch_size, shuffle=False)
    val_dataset_deform = torch.utils.data.TensorDataset(torch.Tensor(data_deform[int(len(data_deform)*0.99):]))
    val_loader_deform = torch.utils.data.DataLoader(val_dataset_deform, batch_size=batch_size, shuffle=False)
    test_dataset_deform = torch.utils.data.TensorDataset(torch.Tensor(test_data_deform))
    test_loader_deform = torch.utils.data.DataLoader(test_dataset_deform, batch_size=batch_size, shuffle=False)
    # output-onshore
    train_dataset_out = torch.utils.data.TensorDataset(torch.Tensor(data_out[0:int(len(data_out)*0.99)]))   
    train_loader_out = torch.utils.data.DataLoader(train_dataset_out, batch_size=batch_size, shuffle=False)
    val_dataset_out = torch.utils.data.TensorDataset(torch.Tensor(data_out[int(len(data_out)*0.99):]))
    val_loader_out = torch.utils.data.DataLoader(val_dataset_out, batch_size=batch_size, shuffle=False)
    test_dataset_out = torch.utils.data.TensorDataset(torch.Tensor(test_data_out))
    test_loader_out = torch.utils.data.DataLoader(test_dataset_out, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    print('Training coupled Autoencoder:with offshore channels:',channels_off, 'and deform/on shore channels:',channels_on)

    if couple_epochs[0] == None :
        offshore_model = torch.load(f"{MLDir}/model/{reg}/out/model_offshore_ch_{channels_off}_minepoch_{train_size}.pt")
        onshore_model = torch.load(f"{MLDir}/model/{reg}/out/model_onshore_ch_{channels_on}_minepoch_{train_size}.pt")
        deform_model = torch.load(f"{MLDir}/model/{reg}/out/model_deform_ch_{channels_on}_minepoch_{train_size}.pt")
    elif couple_epochs[0] != None and couple_epochs[1] != None:
        offshore_model = torch.load(f"{MLDir}/model/{reg}/out/model_offshore_ch_{channels_off}_epoch_{couple_epochs[0]}_{train_size}.pt")
        onshore_model = torch.load(f"{MLDir}/model/{reg}/out/model_onshore_ch_{channels_on}_epoch_{couple_epochs[1]}_{train_size}.pt")
        deform_model = torch.load(f"{MLDir}/model/{reg}/out/model_deform_ch_{channels_on}_epoch_{couple_epochs[0]}_{train_size}.pt")
        
    model = Autoencoder_coupled(offshore_model,onshore_model,deform_model,interface_layers,tune_nlayers)
    model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_epoch_losses = []
    val_epoch_losses = []
    test_epoch_losses = []

    # Train model
    for epoch in range(nepochs):
        train_loss = 0
        val_loss = 0
        test_loss = 0
        for batch_data_in,batch_data_deform,batch_data_out in zip(train_loader_in,train_loader_deform,train_loader_out):
            optimizer.zero_grad()
            batch_data_in = batch_data_in[0].to('cuda')
            batch_data_deform = batch_data_deform[0].to('cuda')
            batch_data_out = batch_data_out[0].to('cuda')
            recon_data = model(batch_data_in,batch_data_deform)
            loss = criterion(recon_data, batch_data_out)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
                
        for batch_data_in,batch_data_deform,batch_data_out in zip(val_loader_in,val_loader_deform,val_loader_out):
            batch_data_in = batch_data_in[0].to('cuda')
            batch_data_deform = batch_data_deform[0].to('cuda')
            batch_data_out = batch_data_out[0].to('cuda')
            recon_data = model(batch_data_in,batch_data_deform)
            vloss = criterion(recon_data, batch_data_out)
            val_loss += vloss.item()

        for batch_data_in,batch_data_deform,batch_data_out in zip(test_loader_in,test_loader_deform,test_loader_out):
            batch_data_in = batch_data_in[0].to('cuda')
            batch_data_deform = batch_data_deform[0].to('cuda')
            batch_data_out = batch_data_out[0].to('cuda')
            recon_data = model(batch_data_in,batch_data_deform)
            tloss = criterion(recon_data, batch_data_out)
            test_loss += tloss.item()

        avg_train_ls = train_loss/len(train_loader_in)
        avg_val_ls = val_loss/len(val_loader_in)
        avg_test_ls = test_loss/len(test_loader_in)

        if verbose:
            print(f'epoch:{epoch},training loss:{avg_train_ls:.5f},val loss:{avg_val_ls:.5f},test loss:{avg_test_ls:.5f}', end='\r')
         
        #append losses to explore overfitting
        train_epoch_losses.append(avg_train_ls)
        val_epoch_losses.append(avg_val_ls)
        test_epoch_losses.append(avg_test_ls)
        
        #save model a sepcific intermediate epoch
        if epoch % 100 == 0 :#and epoch >= 800:
           torch.save(model, f'{MLDir}/model/{reg}/out/model_coupled_off{channels_off}_on{channels_on}_epoch_{epoch}_{train_size}.pt')
        
        #overwrite epochs where val + test loss are the minimum and mark in plot below:
        if epoch == 0:
            min_loss = avg_val_ls + avg_test_ls
            min_epoch = epoch
        elif avg_val_ls + avg_test_ls < min_loss:
            min_loss = avg_val_ls + avg_test_ls
            min_epoch = epoch
            torch.save(model, f'{MLDir}/model/{reg}/out/model_coupled_off{channels_off}_on{channels_on}_minepoch_{train_size}.pt')

        #save loss stats and min val + test loss at last epoch
        if epoch == nepochs-1:
            print(f'epoch:{epoch},training loss:{avg_train_ls:.5f},val loss:{avg_val_ls:.5f},test loss:{avg_test_ls:.5f}', end='\r')
            torch.save(model, f'{MLDir}/model/{reg}/out/model_coupled_off{channels_off}_on{channels_on}_epoch_{epoch}_{train_size}.pt')
    
    print('min loss at epoch:',min_epoch) 

    #plot training/val loss
    plt.plot(train_epoch_losses, color='blue')
    plt.plot(val_epoch_losses, color='red')
    plt.plot(test_epoch_losses, color='green')
    plt.axvline(x=min_epoch, color='black', linestyle='--')
    plt.text(min_epoch, 0.1, min_epoch, fontsize=12)
    plt.legend(['train', 'val', 'test'], loc='upper left')
    plt.title(f"Training loss for Nofold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.savefig(f'{MLDir}/model/{reg}/plot/model_coupled_off{channels_off}_on{channels_on}_{train_size}.png')   
    plt.clf()


def evaluateAE(data_in, #training data offshore
               data_deform, #training data deformation
               data_out, #training data onshore
               model_def, #model feature inputs and outputs
               channels_off = [64,128,256], #channels for offshore(1DCNN)
               channels_on = [64,64], #channels for onshore(fully connected)
               epoch =  None,#selected epoch
               batch_size = 100, #depends on GPU memory
               control_points = [], #control points for evaluation
               threshold = 0.1, #threshold for evaluation
               verbose = False):
    
    train_size = mask_size
    test_size = size
    #read model from file for testing
    if epoch is None:
        model = torch.load(f'{MLDir}/model/{reg}/out/model_coupled_off{channels_off}_on{channels_on}_minepoch_{train_size}.pt')
    else:
        model = torch.load(f'{MLDir}/model/{reg}/out/model_coupled_off{channels_off}_on{channels_on}_epoch_{epoch}_{train_size}.pt') 
    model.eval()

    print('model summary.....')
    # summary(model,[(model_def[0],model_def[1]),(model_def[2])])

    # Test model for final evaluation
    predic = np.zeros(data_out.shape)
    criterion = nn.MSELoss()

    test_dataset_in = torch.utils.data.TensorDataset(torch.Tensor(data_in))
    test_loader_in = torch.utils.data.DataLoader(test_dataset_in, batch_size=batch_size, shuffle=False)
    test_dataset_deform = torch.utils.data.TensorDataset(torch.Tensor(data_deform))
    test_loader_deform = torch.utils.data.DataLoader(test_dataset_deform, batch_size=batch_size, shuffle=False)
    test_dataset_out = torch.utils.data.TensorDataset(torch.Tensor(data_out))
    test_loader_out = torch.utils.data.DataLoader(test_dataset_out, batch_size=batch_size, shuffle=False)

    # Test model
    with torch.no_grad():
        test_loss = 0
        for batch_idx,(batch_data_in,batch_data_deform,batch_data_out) in enumerate(zip(test_loader_in,test_loader_deform,test_loader_out)):
            batch_data_in = batch_data_in[0].to('cuda')
            batch_data_deform = batch_data_deform[0].to('cuda')
            batch_data_out = batch_data_out[0].to('cuda')
            recon_data = model(batch_data_in,batch_data_deform)
            loss = criterion(recon_data, batch_data_out)
            test_loss += loss.item()
            predic[batch_idx*batch_size:(batch_idx+1)*batch_size] = recon_data.cpu().numpy()
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
    plt.savefig(f'{MLDir}/model/{reg}/plot/model_coupled_off{channels_off}_on{channels_on}_{train_size}_maxdepth_testsize{test_size}.png')

    #first calculate location index of control points for given lat and lon
    locindices = get_idx_from_latlon(control_points)

    #evaluation table
    eve_perf = []
    true_list = []
    pred_list = []
    er_list = []

    #mse_val,r2_val,pt_er,KCap,Ksmall,truecount,predcount
    test_ids = np.loadtxt(f'{MLDir}/data/events/shuffled_events_{mode}_{test_size}.txt',dtype='str')
    for eve_no,eve in enumerate(test_ids):
        scores = calc_scores(data_out[eve_no,:], predic[eve_no,:],locindices,threshold=0.1)
        eve_perf.append([scores[0],scores[1],#scores[3],scores[4], #mse,r2,KCap,Ksmall
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

    #print overall performance metrics for whole training exercise and evaluation work : mseoverall, K, k, r2maxdepth
    # Calculate mseoverall and r2maxdepth
    mseoverall = mean_squared_error(data_out, predic)
    print(f"mseoverall: {mseoverall:.4f}")
    print(f"r2maxdepth: {r2maxdepth:.4f}")
    #TODO: add per event evaluation for discovery and analysis

    #plot error at each location
    print('plotting error at each control points')
    plt.figure(figsize=(15, 30))
    #add to main plot the mse and r2 to the plot at the top
    plt.suptitle(f"mseoverall: {mseoverall:.5f},r2maxdepth: {r2maxdepth:.5f}_testsize: {test_size}",fontsize=25)
    #error charts
    for i in range(len(locindices)):
        plt.subplot(6,2,i+1)
        plt.hist(er_list[er_list[:,i]!=0,i],bins=5)
        #set x axis to be the same for all subplots
        plt.xlim(-2,2)
        #calculate hit and mis for each location based on depth of true and prediction
        #events crossing the threshold of 0.2 are considered flooded
        neve = np.count_nonzero(true_pred_er[:,i]>threshold) # no of flooded grid points in the event
        # print(neve)
        #true positive: true>0.2 and pred>0.2
        TP = np.count_nonzero((true_pred_er[:,i]>threshold) & (true_pred_er[:,i+4]>threshold))/(neve)
        TN = np.count_nonzero((true_pred_er[:,i]<threshold) & (true_pred_er[:,i+4]<threshold))/(len(true_pred_er[:,i])-neve)
        FP = np.count_nonzero((true_pred_er[:,i]<threshold) & (true_pred_er[:,i+4]>threshold))/(len(true_pred_er[:,i])-neve)
        FN = np.count_nonzero((true_pred_er[:,i]>threshold) & (true_pred_er[:,i+4]<threshold))/(neve)
        plt.title(f"Control Location:{i+1},No of flood events:{neve}/len:{len(true_pred_er[:,i])}")
        plt.text(0.78, 0.9, f" TP: {TP:.2f}, TN: {TN:.2f}", horizontalalignment='center',verticalalignment='center', transform=plt.gca().transAxes,fontsize=12)
        plt.text(0.78, 0.75, f"FP: {FP:.2f}, FN: {FN:.2f}", horizontalalignment='center',verticalalignment='center', transform=plt.gca().transAxes,fontsize=12)
        plt.xlabel('Error')
        plt.ylabel('Count')
    plt.savefig(f'{MLDir}/model/{reg}/plot/model_coupled_off{channels_off}_on{channels_on}_error_testsize{test_size}.png')

def calc_scores(true,pred,locindices,threshold=0.1): #for each event
    #only test where there is significant flooding
    true[true<threshold] = 0
    pred[pred<threshold] = 0
    mse_val = mean_squared_error(true,pred)
    r2_val = r2_score(true,pred)
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

    return mse_val,r2_val,true[locindices],pred[locindices],pt_er 

def get_idx_from_latlon(locations):
    train_size = mask_size
    firstevent = np.loadtxt(f'{MLDir}/data/events/sample_events53550.txt',dtype='str')[0]
    D_grids = xr.open_dataset(f'{SimDir}/{firstevent}/{reg}_flowdepth.nc')
    zero_mask = np.load(f'{MLDir}/data/processed/zero_mask_{reg}_{train_size}.npy')
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
        ppeaks, _ = scipy.signal.find_peaks(ts[:,g], height=0.05,distance=10)
        npeaks, _ = scipy.signal.find_peaks(-ts[:,g], height=0.05,distance=10)

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
                waveperiod = (ppeaks[1]-ppeaks[0])
        elif polarity == '-1':
            if len(npeaks)==1:
                waveperiod = 0
            else:
                waveperiod = (npeaks[1]-npeaks[0])

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
 
    return df['period']

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
        print('Sampling',samples_per_bin, 'events in bin',bin,'from',len(events_in_bin),'events')
        
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
        else:
            events_in_bin_copy.loc[:, 'norm_wt'] = events_in_bin[importance_column] / np.sum(events_in_bin[importance_column])  

        if samples_per_bin > 0 and np.sum(events_in_bin_copy['norm_wt']) > 0:
            #sample events from this bin weighted by norm_wt
            if len(events_in_bin) <= samples_per_bin:
                print('Less scenario to sample in this bin',bin,' -- need to be careful using samples will use sample with replacement')
                sampled_ids = np.random.choice(events_in_bin['id'],
                                                size=samples_per_bin,
                                                p=events_in_bin_copy['norm_wt'],
                                                replace=True,
                                                )
            else:
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

def sample_events(wt_para = 'gridcount', #'LocationCount', 'mean_prob', 'importance', 'uniform_wt', 'gridcount'
                  samples_per_bin = 15,
                  bin_splits = 12):
    if reg == 'SR':
        gaugeno = str(54)
    elif reg == 'CT':
        gaugeno = str(41)

    data = pd.read_csv(MLDir + f'/data/info/sampling_input_{reg}_{gaugeno}.csv')

    'split sampling in two steps for event type 0 and 1 then merge'
    sample_step0 = sample_train_events(data.groupby('event_type').get_group(0),
                                    importance_column=wt_para,
                                    samples_per_bin=samples_per_bin,
                                    bin_def = np.append(np.linspace(0,3.5,bin_splits), 99))

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
    plt.savefig(MLDir + f'/model/{reg}/plot/sampledist_events{str(sample_len)}_{reg}_{gaugeno}.png', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(15,10))
    ax.title.set_text('Sampled events by: ' + wt_para)
    ax = plt.scatter(sample_test[sample_test['event_type']==1]['lon'], sample_test[sample_test['event_type']==1]['lat'], alpha=0.75, label='far',s=4)
    ax = plt.scatter(sample_test[sample_test['event_type']==0]['lon'], sample_test[sample_test['event_type']==0]['lat'], alpha=0.75, label='near', s=5)
    plt.legend(loc='upper right')
    plt.xlim(10, 35)
    plt.ylim(30, 43)
    ax = plt.gca()
    cx.add_basemap(ax,crs='EPSG:4326', source=cx.providers.Stamen.TonerLite)
    plt.savefig(MLDir + f'/model/{reg}/plot/samplemap_events{str(sample_len)}_{reg}_{gaugeno}.png', bbox_inches='tight')

    #save list of ids to txt file
    path = MLDir + f'/data/events/sample_events{str(sample_len)}_{reg}_{gaugeno}.txt'
    sample_test['id'].unique().to_csv(path, header=False, index=None)
    
    return sample_len, path