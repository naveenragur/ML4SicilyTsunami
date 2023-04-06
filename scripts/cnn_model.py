import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random
import copy
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class CNN(nn.Module):
    def __init__(self, 
                ninput,
                noutput,
                arch='riku',
                set_dropout=True):
        super().__init__()

        self.set_dropout = set_dropout

        if set_dropout==True:
            self.dropout = nn.Dropout(p=0.2)

        enc = [32, 32, 32]
        dec = [0, 0, 0]
        self.enc_channels_list = enc
        self.dec_channels_list = dec

        # encoder
        self.conv1 = nn.Conv1d(ninput, enc[0], 3, padding=1)
        self.conv2 = nn.Conv1d(enc[0], enc[1], 3, padding=1)
        self.conv3 = nn.Conv1d(enc[1], enc[2], 3, padding=1)

        # latent space and activation
        self.makeflat = nn.Flatten()
        self.pool = nn.MaxPool1d(4,4)
        self.batchnorm = nn.BatchNorm1d(32)
        self.dense = nn.Linear(32*4, noutput) #out channel dim X 
        self.relu = nn.LeakyReLU(negative_slope=0.5) 
                
    def forward(self, x):
        set_dropout = self.set_dropout

        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.batchnorm(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        # print(x.shape)
        if set_dropout==True:
            x = self.dropout(x)
        x = self.makeflat(x)
        x = self.dense(x)
        
        return x

class MaxGaugeModel():
    def __init__(self,
                 data_path='./_data',
                 data_name='riku',
                 model_name='riku',
                 ndata=490,
                 gauges=[5832, 6042],
                 npts_in=1024,
                 npts_out=1,
                 device_type='cpu'):

        self.data_path = data_path

        if not os.path.exists('_output'):
            os.mkdir('_output')
        self.output_path = '_output'
        
        self.data_fname = None
        self.data_name = data_name

        # data shapes
        if ndata == None:
            runno_filepath = os.path.join(self.data_path, '_runno.txt'.format(data_name))
            self.ndata = len(open(runno_filepath).readlines(  ))
        else:
            self.ndata = ndata 
        self.gauges = gauges  
        self.ngauges = len(gauges)
        self.npts_in = npts_in
        self.npts_out = npts_out

        self.shuffled = None
        self.shuffle_seed = 0
        self.init_weight_seed = 10

        self.model_name = model_name
        self.model_ninput = None
        self.model_noutput = None

        self.input_gauges_bool = None
        self.output_gauges_bool = None

        self.device = device_type

        self.shuffled_batchno = False
        self.data_batches = None

        self.use_Agg = True
        # set dpi for plt.savefig
        self._dpi = 300

    def load_data(self,
                  batch_size=20,
                  data_fname=None):
        '''
        load interpolated gauge data 

        set data_fname to designate stored data in .npy format
        
        '''

        device = self.device

        if data_fname == None:
            fname = self.model_name + '.npy'
            data_fname = os.path.join(self.data_path, fname)

        data_all = np.load(data_fname)
        data_name = self.data_name
        
        # load shuffled indices 
        fname = os.path.join(self.data_path,'{:s}_train_index.txt'.format(data_name))
        train_index = np.loadtxt(fname).astype(np.int)
        self.train_index =  train_index
                
        fname = os.path.join(self.data_path,'{:s}_test_index.txt'.format(data_name))
        test_index = np.loadtxt(fname).astype(np.int)
        self.test_index = test_index

        data_train = data_all[train_index, : , :]
        data_test  = data_all[test_index, : , :]

        data_out = data_train[:,1,:].max(axis=1)
        fname = '{:s}_max_output_gauge_obs_train.txt'.format(data_name)
        save_fname = os.path.join('_output', fname)
        np.savetxt(save_fname, data_out)

        data_out = data_test[:,1,:].max(axis=1)
        fname = '{:s}_max_output_gauge_obs_test.txt'.format(data_name)
        save_fname = os.path.join('_output', fname)
        np.savetxt(save_fname, data_out)

        # create a list of batches for training, test sets
        data_train_batch_list = []
        data_test_batch_list = []

        self.batch_size = batch_size
        for i in np.arange(0, data_train.shape[0], batch_size):
            data0 = data_train[i:(i + batch_size), :, :]
            data0 = torch.tensor(data0, dtype=torch.float32).to(device)
            data_train_batch_list.append(data0)

        for i in np.arange(0, data_test.shape[0], batch_size):
            data0 = data_test[i:(i + batch_size), :, :]
            data0 = torch.tensor(data0, dtype=torch.float32).to(device)
            data_test_batch_list.append(data0)

        self.nbatches_train = len(data_train_batch_list)
        self.nbatches_test  = len(data_test_batch_list)

        self.ndata_train = sum([data0.shape[0] for 
                           data0 in data_train_batch_list])
        self.ndata_test  = sum([data0.shape[0] for 
                           data0 in data_test_batch_list])

        self.data_train_batch_list = data_train_batch_list
        self.data_test_batch_list = data_test_batch_list
        
        self.data_fname = data_fname

    def train_ensemble(self,
                        nensemble=25,
                        torch_loss_func=nn.L1Loss,
                        torch_optimizer=optim.Adam,
                        nepochs=500,
                        save_interval=None,
                        input_gauges=None,
                        weight_decay=0.0,
                        lr=0.0001):
            '''
            Train autoencoder ensembles and pickles them in the output dir

            Parameters
            ----------
            nensemble :
                number of models in the ensembles

            torch_loss_func :
                pytorch loss function, default is torch.nn.MSELoss

            torch_optimizer :
                pytorch loss function, default is optim.Adam

            gauges : list
            list of all gauges 
            
            input_gauges : list
            gauges to use as inputs, sublist of gauges
           

            '''

            # store training hyper-parameters
            self.nensemble = nensemble
            self.nepochs = nepochs
            self.torch_loss_func = torch_loss_func.__name__
            self.torch_optimizer = torch_optimizer.__name__

            model_name = self.model_name
            batch_size = self.batch_size

            # select hardware
            device = self.device

            #batch info
            data_train_batch_list = self.data_train_batch_list
            data_test_batch_list = self.data_test_batch_list
            nbatches_train = self.nbatches_train
            nbatches_test = self.nbatches_test
            ndata_train = self.ndata_train
            ndata_test = self.ndata_test
 
            # set save interval: save model every ``save_interval`` epochs or atleast 10
            if save_interval == None:
                save_interval = int(nepochs/10)    

            # set random seed
            init_weight_seed = self.init_weight_seed
            torch.random.manual_seed(init_weight_seed)

            # set output path
            output_path = self.output_path
         
            # set ninputs and noutputs
            if input_gauges == None:
                input_gauges = self.gauges[:1]
            model_ninput = len(input_gauges)

            model_noutput = self.ngauges - model_ninput

            # set gauge bools, indices and data points
            self.input_gauges = input_gauges
            input_gauges_bool = np.array([gauge in input_gauges for gauge in self.gauges])
            self.input_gauges_bool = input_gauges_bool

            ig = np.arange(self.ngauges)[input_gauges_bool]
            og = np.arange(self.ngauges)[~input_gauges_bool]

            self.model_ninput = model_ninput
            self.model_noutput = model_noutput
            self.train_lr = lr

            # save model info
            model = CNN(model_ninput, model_noutput, arch=model_name)
            self.enc_channels_list = model.enc_channels_list
            self.dec_channels_list = model.dec_channels_list
            self.save_model_info()              

            for n_model in range(nensemble):

                # define new model
                model = CNN(model_ninput, model_noutput, arch=model_name)
                model.to(device)

                self.enc_channels_list  = model.enc_channels_list
                self.dec_channels_list = model.dec_channels_list

                # train model
                loss_func = torch_loss_func()
                optimizer = torch_optimizer(model.parameters(), 
                                            lr=lr, 
                                            weight_decay=weight_decay)

                # save loss at epochs
                train_loss_array = np.zeros(nepochs)
                test_loss_array = np.zeros(nepochs)


                for epoch in range(1, nepochs+1):
                    # monitor training loss
                    train_loss = 0.0
                    test_loss = 0.0

                    # train model over each batch
                    for k in range(nbatches_train):
                        #train
                        data0 = data_train_batch_list[k] #a batch of data with shape (batch_size, ngauges, npts)

                        # input is selected gauges and output is all the remaining gauges
                        data_in = data0[:,ig,:].detach().clone()
                        data_out = data0[:,og,:].max(axis=2).values.detach().clone()
                        optimizer.zero_grad()

                        model_out = model(data_in)
                        loss = loss_func(model_out, data_out)

                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        
                        #test
                        if k < nbatches_test:
                            datatest0 = data_test_batch_list[k]
                            datatest_in = datatest0[:,ig,:].detach().clone()
                            datatest_out = datatest0[:,og,:].detach().clone().max(axis=2).values

                            model_out_test = model(datatest_in)
                            tesloss = loss_func(model_out_test, datatest_out)
                            test_loss += tesloss.item()
                            
                    # avg training loss per epoch
                    avg_train_loss = train_loss/nbatches_train
                    train_loss_array[epoch-1] = avg_train_loss 

                    avg_test_loss = test_loss/nbatches_test
                    test_loss_array[epoch-1] = avg_test_loss  

                    # display status
                    msg = '\rensemble no = {:4d}, epoch = {:4d}, trainloss = {:1.8f}, testloss = {:1.8f}'\
                        .format(n_model,epoch,avg_train_loss,avg_test_loss)
                    sys.stdout.write(msg)
                    sys.stdout.flush()

                    fname = '{:s}_train_loss_m{:02d}.npy'\
                            .format(model_name,n_model)
                    save_fname = os.path.join(output_path, fname)
                    np.save(save_fname, train_loss_array)

                    fname = '{:s}_test_loss_m{:02d}.npy'\
                            .format(model_name,n_model)
                    save_fname = os.path.join(output_path, fname)
                    np.save(save_fname, test_loss_array)

                    if ((epoch) % save_interval) == 0:
                        # save intermediate model
                        fname ='{:s}_model_m{:04d}_e{:04d}.pkl'\
                                .format(model_name, n_model, epoch)
                        save_fname = os.path.join(output_path, fname)
                        torch.save(model, save_fname)

    def save_model_info(self):

        import pickle

        info_dict = [self.data_name,
                    self.ndata,
                    self.npts_in,
                    self.npts_out,
                    self.batch_size,
                    self.nbatches_train,
                    self.nbatches_test,
                    self.ndata_train,
                    self.ndata_test,
                    self.nepochs,
                    self.nensemble,
                    self.gauges,
                    self.ngauges,
                    self.input_gauges,
                    self.input_gauges_bool,
                    self.model_ninput,
                    self.model_noutput,
                    self.data_path,
                    self.output_path,
                    self.shuffled,
                    self.shuffle_seed,
                    self.init_weight_seed,
                    self.shuffled_batchno,
                    self.train_index,   
                    self.test_index,
                    self.model_name,
                    self.data_fname,
                    self.enc_channels_list,
                    self.dec_channels_list,
                    self.train_lr,
                    self.torch_optimizer,
                    self.torch_loss_func,
                    self.device]
        
        fname = '{:s}_info.pkl'.format(self.model_name)
        save_fname = os.path.join(self.output_path, fname)
        pickle.dump(info_dict, open(save_fname,'wb'))

        fname = '{:s}_info.txt'.format(self.model_name)
        save_txt_fname = os.path.join(self.output_path, fname)

        def _append(outstring, x):
            if type(x) is int:
                outstring += '\n {:d}'.format(x)
            elif type(x) is str:
                outstring += '\n {:s}'.format(x)
            return outstring

        with open(save_txt_fname, mode='w') as outfile:
            for k, item in enumerate(info_dict):
                outstring = ' --- {:d} --- '.format(k)
                if type(item) is list:
                    for x in item:
                        _append(outstring, x)
                else:
                    _append(outstring, item)
                outfile.write(outstring)
      
    def load_model(self,model_name,device=None):

        import pickle

        # load model info
        fname = '{:s}_info.pkl'.format(model_name)
        load_fname = os.path.join(self.output_path, fname)
        info_dict = pickle.load(open(load_fname,'rb'))

        [self.data_name,
        self.ndata,
        self.npts_in,
        self.npts_out,
        self.batch_size,
        self.nbatches_train,
        self.nbatches_test,
        self.ndata_train,
        self.ndata_test,
        self.nepochs,
        self.nensemble,
        self.gauges,
        self.ngauges,
        self.input_gauges,
        self.input_gauges_bool,
        self.model_ninput,
        self.model_noutput,
        self.data_path,
        self.output_path,
        self.shuffled,
        self.shuffle_seed,
        self.init_weight_seed,
        self.shuffled_batchno,
        self.train_index,   
        self.test_index,
        self.model_name,
        self.data_fname,
        self.enc_channels_list,
        self.dec_channels_list,
        self.train_lr,
        self.torch_optimizer,
        self.torch_loss_func,
        self.device] = info_dict

        if device != None:
            self.device = device

        # load data
        self.load_data(batch_size=self.batch_size)

    def predict_dataset(self, 
                        epoch, 
                        model_list=None,
                        data_subset_list=['train', 'test'],
                        device='cpu'):
        r"""
        Predict all of the data set, both training and test sets

        Parameters
        ----------

        epoch : int
            use model after training specified number of epochs

        device : {'cpu', 'cuda'}, default 'cpu'
            choose device for PyTorch modules

        data_subset_list : list, default ['train', 'test']
            list of data subset names

        Notes
        -----
        the prediction result is stored as binary numpy arrays in 
        the output directory

        """
        batch_size = self.batch_size
        model_name = self.model_name

        # load data and data dimensions
        if self.data_batches == None:
            self.load_data(batch_size=batch_size)

        # if model_list is not provided, predict for all models in ensemble
        if model_list == None:
            model_list = [i for i in range(self.nensemble)]

                                      
        get_eval_model = self.get_eval_model
        input_gauges_bool = self.input_gauges_bool
        
        npts_in = self.npts_in 
        npts_out = self.npts_out
        model_noutput = self.model_noutput

        for data_subset in data_subset_list:           
            if data_subset == 'train':
                data_batch_list = self.data_train_batch_list
                ndata = self.ndata_train
                nbatches = self.nbatches_train
            elif data_subset == 'test':
                data_batch_list = self.data_test_batch_list
                ndata = self.ndata_test
                nbatches = self.nbatches_test

            for n in model_list:
                fname = '{:s}_pred_{:s}-input_m{:04d}_e{:04d}.dat'\
                        .format(model_name,data_subset, n, epoch)
                save_fname = os.path.join('_output', fname)

                pred_all = np.memmap(save_fname,
                                    mode='w+',
                                    dtype=float,
                                    shape=(ndata, model_noutput, npts_out))

                model = get_eval_model(n, epoch, device=device)

                kb = 0 #event counter

                for k in range(nbatches): 
                    msg = '\repoch{:6d},=data_subset={:s}, model={:6d}, batch={:6d}'.format(epoch,data_subset, n, k)
                    # sys.stdout.write(msg)
                    # sys.stdout.flush()
                    print(msg)

                    # setup input data
                    data0 = data_batch_list[k].to(device)
                    data_in = data0[:,input_gauges_bool,:].detach().clone()

                    batch_size = data_in.shape[0] #might not be exactly batch_size in the last batch

                    # evaluate model
                    with torch.no_grad():
                        model_out = model(data_in)

                    # collect predictions
                    ke = kb + batch_size
                    pred_all[kb:ke, :, :] = model_out.detach().numpy().reshape(batch_size, model_noutput, npts_out) 
                    kb = ke

                # save predictions as text file
                fname = '{:s}_pred_{:s}-input_m{:04d}_e{:04d}.txt'\
                        .format(model_name,data_subset, n, epoch)
                save_fname = os.path.join('_output', fname)
                np.savetxt(save_fname, pred_all[:, 0, 0])
                
    def predict_historic(self, 
                        epoch, 
                        model_list=None,
                        history_list=None,
                        device='cpu'):
        r"""
        Predict all of the data set, both training and test sets

        Parameters
        ----------

        epoch : int
            use model after training specified number of epochs

        device : {'cpu', 'cuda'}, default 'cpu'
            choose device for PyTorch modules

        hist_list : list, default if None use hardcoded list
            list of historic event names

        Notes
        -----
        the prediction result is stored as binary numpy arrays in 
        the output directory

        """
        model_name = self.model_name
    
        # if model_list is not provided, predict for all models in ensemble
        if model_list == None:
            model_list = [i for i in range(self.nensemble)]

        get_eval_model = self.get_eval_model
        gauges = self.gauges
        ngauges = self.ngauges
        input_gauges_bool = self.input_gauges_bool

        npts_in = self.npts_in 
        npts_out = self.npts_out
        model_ninput = self.model_ninput
        model_noutput = self.model_noutput

        if history_list == None:
            # hist_name_list = ['FUJI2011_42']
            # hist_name_list = ['SL_' + "{0:04}".format(n) for n in range(14)]
            hist_name_list = ['FUJI2011_42','NANKAI2022','SANRIKU1896','SANRIKU1933','TOKACHI1968','YAMAZAKI2018_TPMOD',
            'SatakeMiniLowerSoft','SatakeMiniUpper','SatakeMiniUpperSoft','SatakeMiniLower']

        ndata=len(hist_name_list)   

        gauge_data = np.zeros((ndata, ngauges, npts_in)) #data points to recreate time series
        etamax_obs = np.zeros((ndata, ngauges))
        for i,hist_name in enumerate(hist_name_list):
            gauge_path = os.path.join('/mnt/data/nragu/Tsunami/Tohoku/_results/_output6hrs', hist_name)
            for k, gauge in enumerate(gauges):
                fname = 'gauge{:05d}.txt'.format(gauge)
                load_fname = os.path.join(gauge_path, fname)
                raw_gauge = np.loadtxt(load_fname, skiprows=3)
                dz=raw_gauge[1,5]
                t = raw_gauge[:, 1]
                eta = raw_gauge[:, 5]-dz
                etamax_obs[i, k] = eta.max()   

                if k == 0:
                    # set prediction time-window
                    t_init = np.min(t[np.abs(eta) > 0.1])
                    t_final = t_init + 4.0 * 3600.0
                    t_unif = np.linspace(t_init, t_final, npts_in) #data points to recreate time series    

                # interpolate to uniform grid on prediction window
                gauge_data[i, k, :] = np.interp(t_unif, t, eta)

        # setup input data
        output_fname = os.path.join(self.data_path,
                                '{:s}_{:s}.npy'.format(model_name,'historic'))
        np.save(output_fname, gauge_data)
        data0 = torch.tensor(gauge_data, dtype=torch.float32).to(device)
        data_in = data0[:,input_gauges_bool,:].detach().clone()

        # predict
        for n in model_list:
            fname = '{:s}_pred_{:s}-input_m{:04d}_e{:04d}.dat'\
                    .format(model_name,'historic', n, epoch)
            save_fname = os.path.join('_output', fname)

            pred_all = np.memmap(save_fname,
                                mode='w+',
                                dtype=float,
                                shape=(ndata, model_noutput, npts_out))
        
            model = get_eval_model(n, epoch, device=device)

            # evaluate model
            with torch.no_grad():
                model_out = model(data_in)

            pred_all[:,:,:] = model_out.detach().detach().numpy().reshape(ndata, model_noutput, npts_out) 
       
            # save predictions as text file
            fname = '{:s}_pred_{:s}-input_m{:04d}_e{:04d}.txt'\
                    .format(model_name,'historic', n, epoch)
            save_fname = os.path.join('_output', fname)
            np.savetxt(save_fname, pred_all[:, 0, 0])

            # save true etamax from raw data
            etamax_obs_out = etamax_obs[:,~input_gauges_bool]
            save_fname = '_output/{:s}_max_output_gauge_obs_{:s}.txt'.format(model_name,'historic')
            np.savetxt(save_fname, etamax_obs_out)

    def get_eval_model(self, n, epoch, device='cpu'):
        r"""
        Returns autoencoder model in evaluation mode

        Parameters
        ----------
        n : int
            model number in the ensemble
        
        epoch : int
            use model saved after specified number of epochs

        device : {'cpu', 'cuda'}, default 'cpu'
            choose device for PyTorch modules

        Returns
        -------
        model : NN module
            Conv1d module in evaluation mode
        """

        model_name = self.model_name

        # load stored cnn 
        fname = '{:s}_model_m{:04d}_e{:04d}.pkl'\
                .format(model_name, n, epoch)

        load_fname = os.path.join('_output', fname)
        model = torch.load(load_fname,map_location=torch.device(device))

        model.eval()

        return model

    def _get_pred(self, m, e, data_subset='test'):

        model_name = self.model_name
    
        ndata_train = self.ndata_train
        ndata_test  = self.ndata_test 

        if data_subset == 'train':
            ndata = ndata_train
        elif data_subset == 'test':
            ndata = ndata_test
        elif data_subset == 'historic':
            ndata = 10

        model_noutput = self.model_noutput
        npts_out = self.npts_out

        fname = '{:s}_pred_{:s}-input_m{:04d}_e{:04d}.dat'\
                .format(model_name, data_subset, m, e)
        save_fname = os.path.join('_output', fname)

        pred = np.memmap(save_fname,
                        mode='r+',
                        dtype=float,
                        shape=(ndata, model_noutput, npts_out))
        return pred
    
    def make_pred_plot(self, 
                       data_subset_list=['test', 'train', 'historic'],
                       model_list=None,
                       epochs_list=[300],
                       gauge_labels=None,
                       tf_gauge=4.0):
        r'''
        Make prediction plots (run pred_dataset and historic first)

        '''

        import matplotlib.pyplot as plt

        model_noutput = self.model_noutput

        model_name = self.model_name
        npts_in = self.npts_in
        npts_out = self.npts_out
        model_noutput = self.model_noutput
        get_pred = self._get_pred

        t0 = np.linspace(0.0,tf_gauge,npts_in+1)
        tgauge = 0.5*(t0[1:] + t0[:-1])

        if not os.path.exists('_plots'):
            os.mkdir('_plots')

        if model_list == None:
            model_list = [i for i in range(self.nensemble)]

        if type(gauge_labels) == type(None):
            gauge_labels = ['{:d}'.format(i) for i in range(model_noutput)]

        
      
        ndata_train = self.ndata_train
        ndata_test  = self.ndata_test 
               
        input_gauges_bool = self.input_gauges_bool

        #load sim data
        fname = self.model_name + '.npy'
        hname = self.model_name + '_historic.npy'
        data_all = np.load(os.path.join(self.data_path, fname))
        data_hist =np.load(os.path.join(self.data_path, hname))
        data_name = self.data_name
        
        # load shuffled indices 
        fname = os.path.join(self.data_path,'{:s}_train_index.txt'.format(data_name))
        train_index = np.loadtxt(fname).astype(np.int)
        self.train_index =  train_index
                
        fname = os.path.join(self.data_path,'{:s}_test_index.txt'.format(data_name))
        test_index = np.loadtxt(fname).astype(np.int)
        self.test_index = test_index
     
        #iterate over data subsets
        for data_subset in data_subset_list:
            if data_subset == 'train':
                ndata = ndata_train
                data = data_all[train_index, : , :]
                
            elif data_subset == 'test':
                ndata = ndata_test
                data  = data_all[test_index, : , :]
            
            elif data_subset == 'historic':
                ndata = data_hist.shape[0]
                data  = data_hist
            
            #iterate over epochs
            for e in epochs_list:
            
                etamax_pred = np.zeros((ndata, model_noutput))
                etamax_obs = np.zeros((ndata, model_noutput))
                etamax_pred_sd = np.zeros((ndata, model_noutput))

                plot_path = '_plots/e{:04d}'.format(e)
                if not os.path.exists(plot_path):
                    os.mkdir(plot_path)
                
                #iterate over every event                                
                for i in range(ndata):
                    #obs
                    obs_out = data[i,~input_gauges_bool,:] #observed output gauge 

                    #pred
                    pred_all = []
                    for m in model_list:
                        pred = get_pred(m, e, data_subset=data_subset)
                        pred_all.append(pred)
                    
                    pred_all = np.array(pred_all)
                    fig, axes = plt.subplots(figsize=(10, 3*model_noutput), 
                                                nrows=model_noutput)
                    
                    #iterate over output gauges
                    for og in range(model_noutput):
                        if model_noutput == 1:
                            ax = axes                         
                        else:
                            ax = axes[og]
                        
                        ax.cla()
                        mean_pred = np.mean(pred_all[:, i, og], axis=0) #mean of all model ensembles
                        std_pred = np.std(pred_all[:, i, og], axis=0) #std of all model ensembles

                        etamax_pred[i, og] = mean_pred #max each event and gauge
                        etamax_pred_sd[i, og] = std_pred #max each event and gauge
                        etamax_obs[i, og] = obs_out[og, :].max() #max each event and gauge
                        
                        title='{:s} {:4d}'.format(data_subset, i)
                        ax.set_title(title)
                        ax.fill_between(tgauge, 
                                        mean_pred - 2.0*std_pred, 
                                        mean_pred + 2.0*std_pred, 
                                        color='b', 
                                        label='ML Pred $\pm$ 2std', 
                                        alpha=0.2)
                        ax.plot(tgauge, 
                                obs_out[og, :], 
                                'k--', 
                                label='obs {:d}'.format(og))

                        ax.set_xlabel('hours since threshold passed') 
                        ax.set_ylabel('$\eta$ (m)')
                        ax.legend()
                        fig.tight_layout()

                        fname = '{:s}_pred_{:s}_r{:04d}_e{:04d}.png'\
                            .format(model_name, data_subset, i, e)
                        
                        ffname = os.path.join(plot_path, fname)
                        if data_subset == 'historic':
                            fig.savefig(ffname, dpi=300,facecolor="w")
                        plt.close(fig)

                        sys.stdout.write('\r {:s}'.format(fname))
                        sys.stdout.flush()
                
                np.savetxt(os.path.join(plot_path, 'pred' + data_subset + 'etamax.txt'), etamax_pred)
                np.savetxt(os.path.join(plot_path, 'pred' + data_subset + 'etamax_sd.txt'), etamax_pred_sd)
                np.savetxt(os.path.join(plot_path, 'obs' + data_subset + 'etamax.txt'), etamax_obs)

                for og in range(model_noutput):
                    fig_pvo, ax_pvo = plt.subplots(figsize=(5,4))
                    
                    ax_pvo.plot(etamax_obs[:, og], 
                                etamax_pred[:, og], 
                                'b.', markersize=4)

                    ax_pvo.errorbar(etamax_obs[:, og], 
                                    etamax_pred[:, og],
                                    yerr=etamax_pred_sd[:,og],
                                    fmt='none')

                    vmax = max(max(etamax_pred[:, og]),
                                max(etamax_obs[:, og]))*1.5

                    ax_pvo.set_aspect('equal')
                    ax_pvo.plot([0, vmax], [0, vmax], 'k--')
                    ax_pvo.set_ylabel('ML Pred')
                    ax_pvo.set_xlabel('GC Sim')
                    ax_pvo.set_xlim([0, vmax])
                    ax_pvo.set_ylim([0, vmax])
                    score =str(round(r2_score(etamax_obs[:, og], etamax_pred[:, og]), 3))
                    ax_pvo.text(5, 10, '$R^2 Performance$:' + score, fontsize=8)

                    fname = '{:s}_etamax_{:s}_g{:04d}_e{:04d}.png'.format(model_name, data_subset, og, e)
                    ffname = os.path.join(plot_path, fname)
                    fig_pvo.savefig(ffname, dpi=300,facecolor="w")
                    plt.close(fig_pvo)
            print('\nSet {:s} accuracy: {:s}'.format(data_subset,score))

def interp_gcdata(npts=1024, #1024 observations for 4 hours of data ie ~ 1 per 14 seconds or 4 per minute
                  nruns=771, #number of runs in the simulation results folder to use
                  gauge_path_format= \
                          '/mnt/data/nragu/Tsunami/Tohoku/_results/_output_SLAB/SL_{:04d}/gauge{:05d}.txt',
                  filter_path= \
                          '/mnt/data/nragu/Tsunami/Tohoku/gis/slab_dtopo_list2run4footprint.csv',
                            #slab_dtopo_list2run4footprint is the list of runs to filter some of the runs
                  gauges=[5832, 6042],
                  gauge_data_cols=[1, 5], #time and surface elevation are stored in the 2nd and 6th columns
                  skiprows=3, #skip the first 3 rows of the gauge data
                  thresh_vals=[0.1, 0.5], #offshore and nearshore threshold
                  win_length=4*3600.0,  #filter with min time window
                  data_path='_data', #path to save the processed data in the current directory
                  dataset_name='riku', #name of the dataset
                  make_plots=False, #make plots of the gauge data
                  make_statsplot=False, #make stats [eve_no,mag,filteruse,peak value, peak time]
                  use_Agg=True): #use Agg backend for matplotlib to write files without a display
    '''
    Process GeoClaw gauge output and output time-series interpolated on a
    uniform grid with npts grid pts.


    Parameters
    ----------

    npts : int, default 1024
        no of pts on the uniform time-grid

    skiprows : int, default 3
        number of rows to skip reading GeoClaw gauge output

    nruns : int, default 771 scenarios
        total number of geoclaw runs

    gauge_path_format : string, default  'SL_{:04d}/gauge{:05d}'
        specify sub-directory format of geoclaw output, for example
            'SL_{:04d}/gauge{:05d}'
        the field values will contain run and gauge numbers

    gauges : list of ints, default [5832,6024] for riku
        specify the gauge numbers

    gauge_data_cols : list of ints, default [1, 5]
        designate which columns to use for time and surface elevation
        in the GeoClaw gauge ouput file

    skiprows : int, default 3
        number of rows to skip when reading in the GeoClaw gauge output file

    thresh_vals : list of floats, [0.1, 0.5]
        designate threshold values to impose on the gauge data:
        excludes all runs with:
        abs(eta) < thresh_vals[0] in the first gauge(offshore)
        abs(eta) < thresh_vals[1] in the last gauge(nearshore)

    win_length : float, default 4*3600.0,
        length of the time window in seconds

    data_path : str, default '_data'
        output path to save interpolation results and other information

    dataset_name : str, default 'riku'
        name the dataset

    make_plots : bool, default False
        set to True to generate individual plots for each of the runs

    use_Agg : bool, default False
        set to True to use 'Agg' backend for matplotlib, relevant only when
        kwarg make_plots is set to True
        when used a non-interactive backend that can only write to files

    '''

    #create data path
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    #create some empty arrays filled with zeros
    ngauges = len(gauges)
    data_all = np.zeros((nruns, ngauges, npts))
    i_valid = np.zeros(nruns, dtype=np.bool_)
    t_win_all = np.zeros((nruns, 2))
    run_stat=np.zeros((nruns, ngauges,5)) #5 variables:[eve_no,mag,filteruse,peak value, peak time]

    #set matplotlib backend for plotting
    if use_Agg:
        import matplotlib
        matplotlib.use('Agg')

    if make_plots:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4))

    #load filter list
    filter_list=np.loadtxt(filter_path, skiprows=1, delimiter=',',usecols=(1,2,9))
    
    #start processing geoclaw gauge data by looping over runs
    for i, run_no in enumerate(range(nruns)):
        
        run_data_buffer = [] #empty list to append data for each run
        
        # collect gauge data from each run
        for k, gauge_no in enumerate(gauges):
            gauge_fname = gauge_path_format.format(run_no, gauge_no)
            data = np.loadtxt(gauge_fname, skiprows=skiprows)
            dz = data[1,5] # the deformation at gauge

            sys.stdout.write(
                '\rrun_no = {:6d}, data npts = {:6d}, unif npts = {:d}, gaugeno = {:d}, deformation = {:4.3f}'.format(
                    run_no, data.shape[0], npts,gauge_no, dz))

            # extract relevant columns
            data[:,5] = data[:,5] - dz #correction of local deformation
            run_data_buffer.append(data[1:, gauge_data_cols]) 

            if make_plots: #make plots of the gague data
                if not os.path.isdir("_gaugeplots"):
                    os.mkdir("_gaugeplots")
                ax.cla()
                ax.plot(data[1:, gauge_data_cols[0]], (data[1:, gauge_data_cols[1]]))
                ax.axhline(y = dz, color = 'r', linestyle = '-')
                fig_title = \
                    "run_no {:06d}, gauge {:05d}".format(run_no, gauge_no)
                ax.set_title(fig_title)

                fig_fname = \
                    '_gaugeplots/run_{:06d}_gauge{:05d}.png'.format(run_no, gauge_no)
                fig.savefig(fig_fname, dpi=200)

        valid_data, t_win_lim = _get_window(run_data_buffer,
                                            thresh_vals,
                                            win_length)
       #check if runs is to be included, time window and threshold is satisfied
        if filter_list[i,2] and valid_data: 
            i_valid[i] = 1
        else:
            i_valid[i] = 0

        #interpolate data on a uniform data points and calculate stats[eve_no,mag,filteruse,peak value, peak time,]
        if valid_data :
            t0 = t_win_lim[0]
            t1 = t_win_lim[1]
            t_unif = np.linspace(t0, t1 + 00 * 60.0, npts)

            for k in range(ngauges):
                raw = run_data_buffer[k]
                eta_unif = np.interp(t_unif, raw[:, 0], raw[:, 1])
                data_all[run_no, k, :] = eta_unif
    
                peak = np.amax(np.abs(eta_unif))
                peakelement = (np.argmax(np.abs(eta_unif))*win_length)/(npts*3600)
                stats = [int(filter_list[i,0]),filter_list[i,1],filter_list[i,2],peak,peakelement]
                run_stat[run_no, k, :] = stats

            t_win_all[run_no, 0] = t0 
            t_win_all[run_no, 1] = t1
            sys.stdout.write('   --- valid --- ')
            sys.stdout.flush()
        else:
            pass

    data_all = data_all[i_valid, :, :]
    t_win_all = t_win_all[i_valid, :]
    run_stat = run_stat[i_valid, :, :] 
    
    if make_statsplot: #make plots of the gague data stats
        import matplotlib.pyplot as plt
        fig, sx = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
        if not os.path.isdir("_gaugeplots"):
            os.mkdir("_gaugeplots")

        for k, gauge_no in enumerate(gauges):
            ax=sx
            # frequency plot
            data=run_stat[:,k,0]
            ax.cla()
            ax.hist(data, weights=np.ones_like(data) / len(data))
            fig_title = \
                "Max Amplitude at gauge {:05d}".format(gauge_no)
            ax.set_title(fig_title)
            ax.set_xlabel('Max Amplitude ')
            ax.set_ylabel('Frequency')
            ax.set_xlim(0, 10)

            fig_fname = \
                '_gaugeplots/freq_maxwl_gauge{:05d}.png'.format(gauge_no)
            fig.savefig(fig_fname, dpi=200)
            plt.close(fig)

            #Time of max amplitude
            data=run_stat[:,k,1]
            ax.cla()
            ax.hist(data, weights=np.ones_like(data) / len(data))
            fig_title = \
                "Max Amplitude Time at gauge {:05d}".format(gauge_no)
            ax.set_title(fig_title)
            ax.set_xlabel('Time of Max Amplitude ')
            ax.set_ylabel('Frequency')
            ax.set_xlim(0, 4)

            fig_fname = \
                '_gaugeplots/freq_maxT_gauge{:05d}.png'.format(gauge_no)
            fig.savefig(fig_fname, dpi=200)
            plt.close(fig)
           
            # box plots
            data = pd.DataFrame({'Mag': run_stat[:,k,-1],
            'MaxAmp': run_stat[:,k,0],'MaxAmpTime': run_stat[:,k,1]})
            
            data75=data[data['Mag']==7.5]
            data80=data[data['Mag']==8]
            data85=data[(data['Mag']>=8.5) & (data['Mag'] < 8.75)]
            data90=data[data['Mag'] > 8.75]

            #mag plot - wl
            ax.cla()
            ax.boxplot([data75['MaxAmp'],data80['MaxAmp'],data85['MaxAmp'],data90['MaxAmp']])
            ax.set_xticklabels( ['7.5', '8', '8.5', '9'])

            fig_title = \
                "MaxAmp at diff mag for gauge {:05d}".format(gauge_no)
            ax.set_title(fig_title)
            ax.set_xlabel('Magnitude ')
            ax.set_ylabel('Max Amplitude')


            fig_fname = \
                '_gaugeplots/scatter_mag_maxwl_gauge{:05d}.png'.format(gauge_no)
            fig.savefig(fig_fname, dpi=200)
            plt.close(fig)

            #mag plot - time of peak
            ax.cla()
            ax.boxplot([data75['MaxAmpTime'],data80['MaxAmpTime'],data85['MaxAmpTime'],data90['MaxAmpTime']])
            ax.set_xticklabels( ['7.5', '8', '8.5', '9'])
            fig_title = \
                "Time of Max Amplitude at diff mag for gauge {:05d}".format(gauge_no)
            ax.set_title(fig_title)
            ax.set_xlabel('Magnitude ')
            ax.set_ylabel('Time of Max Amplitude')
  
            fig_fname = \
                '_gaugeplots/scatter_mag_maxtime_gauge{:05d}.png'.format(gauge_no)
            fig.savefig(fig_fname, dpi=200)
            plt.close(fig)

            # save stats as text file
            output_fname = os.path.join(data_path,
                                        '{:05d}_stats.txt'.format(gauge_no))
            np.savetxt(output_fname, run_stat[:,k,:], fmt='%1.3f')

    # save eta 
    output_fname = os.path.join(data_path,
                                '{:s}.npy'.format(dataset_name))
    np.save(output_fname, data_all)

    # save uniform time grid
    output_fname = os.path.join(data_path,
                                '{:s}_t.npy'.format(dataset_name))
    np.save(output_fname, t_win_all)

    # save picked run numbers
    runnos = np.arange(nruns)[i_valid]
    output_fname = os.path.join(data_path,
                                '{:s}_runno.txt'.format(dataset_name))
    np.savetxt(output_fname, runnos, fmt='%d')
#TODO: add a function to also process historic events (e.g. 2011 Tohoku) but might need to rerun everytime new data is available to test hence left it out for now

def _get_window(run_data, thresh_vals, win_length):
    r"""
    Check if data satisfies the requirements: threshold eta at near and offshore

    Parameters
    ----------
    run_data :
        list containing unstructure time reading from the three gauges

    thresh_vals :
        array of size 2 with thresholds for 5832,6042

    win_length :
        length of the prediction window (in seconds)

    Returns
    -------
    valid_data : bool
        True if the run data can be thresholded / windowed properly

    t_win_lim : array
        2-array with beginning and ending time point

    """

    ngauges = len(run_data)
    flag = np.zeros(2, dtype=np.bool_)

    t_win_lim = np.zeros(2)

    # apply threshold to 5832 in default setting(say offshore gauge)
    gaugei_data = run_data[0] #1#index of representative offshore gauge

    t = gaugei_data[:, 0]
    eta = gaugei_data[:, 1]
    t_init = t[0] #time of first reading
    t_final = t[-1] #time of last reading

    i_thresh = (np.abs(eta) >= thresh_vals[0]) #indices of readings above threshold offshore

    if (np.sum(i_thresh) > 0): 
        t_win_init = np.min(t[i_thresh])
        t_win_final = t_win_init + win_length
        t_win_lim[0] = t_win_init
        t_win_lim[1] = t_win_final

        if t_win_final <= t_final:  # check if min time window is available
            flag[0] = True

    # apply threshold to 6042 in default setting(say nearshore gauge)
    gaugei_data = run_data[-1] #-2 #index of representative nearshore gauge found in reverse order incase more than 2 gauges are used

    t = gaugei_data[:, 0]
    eta = gaugei_data[:, 1]

    i_thresh = (np.abs(eta) >= thresh_vals[1]) #indices of readings above threshold nearshore

    if (np.sum(i_thresh) > 0):
        flag[1] = True

    # are both thresholds satisfied?
    valid_data = np.all(flag)
    return valid_data, t_win_lim

def shuffle_dataset(gauges=[5832, 6042],dataset_name='riku', data_path='_data',seed=12345, training_set_ratio = 0.75):
  
    """
    Shuffle interpolated dataset. Run after interp_gcdata()

    Parameters
    ----------

    dataset_name : str, default 'riku'
        set dataset_name, the function requires the file
            '{:s}/{:s}_runno.npy'.format(data_path, dataset_name)

    data_path : str, default '_data'
        output path to save interpolation results and other information

    seed : int, default 12345
        Random seed supplied to np.random.shuffle()

    training_set_ratio : float, default 0.8
    the ratio of total runs to be set as the training set

    Notes
    -----

    Shuffled GeoClaw run numbers or indices are saved in
       '{:s}/{:s}_train_runno.txt'.format(data_path, dataset_name)
       '{:s}/{:s}_train_index.txt'.format(data_path, dataset_name)
       '{:s}/{:s}_test_runno.txt'.format(data_path, dataset_name)
       '{:s}/{:s}_test_index.txt'.format(data_path, dataset_name)

    """

    fname = '{:s}_runno.txt'.format(dataset_name)
    full_fname = os.path.join(data_path, fname)
    shuffled_gc_runno = np.loadtxt(full_fname)

    ndata = len(shuffled_gc_runno)
    shuffled_indices = np.arange(ndata)

    np.random.seed(seed)
    np.random.shuffle(shuffled_indices)
    shuffled_gc_runno = shuffled_gc_runno[shuffled_indices]
    train_index = int(training_set_ratio * ndata)

    # filenames to save
    output_list = [('{:s}_train_runno.txt'.format(dataset_name), # runno till train index for train
                    shuffled_gc_runno[:train_index]),
                   ('{:s}_train_index.txt'.format(dataset_name), # indices till train index for train
                    shuffled_indices[:train_index]),
                   ('{:s}_test_runno.txt'.format(dataset_name), #runno from train index for test
                    shuffled_gc_runno[train_index:]),
                   ('{:s}_test_index.txt'.format(dataset_name), #indices from train index for test
                    shuffled_indices[train_index:])]

    for fname, array in output_list:
        full_fname = os.path.join(data_path, fname)
        np.savetxt(full_fname, array, fmt='%d')
    print("done shuffling!")    

if __name__ == "__main__":
    # threshold, interpolate on window, save data
    interp_gcdata(make_plots=False,make_statsplot=True, use_Agg=True)
    
    # shuffle runs, separate training vs test sets, store data
    shuffle_dataset(data_path='_data', dataset_name='riku',training_set_ratio = 0.8)
       

 