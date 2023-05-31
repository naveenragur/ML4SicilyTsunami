import numpy as np
import pandas as pd

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

try:
    MLDir = os.getenv('MLDir')
    SimDir = os.getenv('SimDir')
    reg = sys.argv[1] #CT or SR
    size = sys.argv[2] #eventset size
except:
    raise Exception("*** Must first set environment variable")

#set seed
np.random.seed(0)

class Autoencoder(nn.Module):
    def __init__(self, ninputs=5,t_len = 480, ch_list = [32,64,96], zdim = 50):
        super(Autoencoder, self).__init__()
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


def trainAE(data,test_data,batch_size,nepochs,lr,n,t,z,channels, verbose = False):
    print('Training Autoencoder:',channels)
    
    # Create PyTorch DataLoader objects
    train_dataset = torch.utils.data.TensorDataset(torch.Tensor(data[0:int(len(data)*0.99)]))   
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = torch.utils.data.TensorDataset(torch.Tensor(data[int(len(data)*0.99):]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_data))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    model = Autoencoder(ninputs = n ,t_len = t, ch_list = channels, zdim = z)
    model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    t_epoch_losses = []
    es_epoch_losses = []
    test_epoch_losses = []

    # Train model
    for epoch in range(nepochs):
        train_loss = 0
        es_losses = 0
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
            es_losses += vloss.item()
        
        for batch_idx, (batch_data,) in enumerate(test_loader):
            batch_data = batch_data.to('cuda')
            recon_data = model(batch_data)
            tloss = criterion(recon_data, batch_data)
            test_loss += tloss.item()

        if verbose:
            print(f'epoch:{epoch} ,training loss:{train_loss/len(train_loader):.5f} ,val loss:{es_losses/len(val_loader):.5f} ,test loss:{test_loss/len(test_loader):.5f}', end='\r')
        if epoch == nepochs:
            print(f'epoch:{epoch} ,training loss:{train_loss/len(train_loader):.5f} ,val loss:{es_losses/len(val_loader):.5f} ,test loss:{test_loss/len(test_loader):.5f}', end='\r')
        
        t_epoch_losses.append(train_loss / len(train_loader))
        es_epoch_losses.append(es_losses / len(val_loader))
        test_epoch_losses.append(test_loss / len(test_loader))
        
        #plot training/val loss
        plt.plot(t_epoch_losses, color='blue')
        plt.plot(es_epoch_losses, color='red')
        plt.plot(test_epoch_losses, color='green')
        plt.legend(['train', 'val', 'test'], loc='upper left')
        plt.title(f"Training loss for Nofold")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale('log')
        plt.savefig(f'{MLDir}/model/{reg}/plot/offshore_loss_ch_{channels}.png')   
        plt.clf()

        #save model
        if epoch % 100 == 0 and epoch >= 800:
            torch.save(model, f'{MLDir}/model/{reg}/out/model_offshore_ch_{channels}_epoch_{epoch}')