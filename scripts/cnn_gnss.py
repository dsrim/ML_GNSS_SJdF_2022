import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

# convolutional neural network
class CNN(nn.Module):
    def __init__(self, 
                 ninput,
                 noutput,
                 arch='sjdf',
                 set_dropout=True):

        super().__init__()

        self.set_dropout = set_dropout

        # TODO: supply a list of channels, instead of the ngauges, ninput pair
        # encoder input 
        if arch=='sjdf':

            enc = [ 64,  64, 128, 128, 128, 256, 256, 512, 512]
            dec = [512, 512, 256, 256, 128, 128,  64,  64]

            self.enc_channels_list = enc
            self.dec_channels_list = dec

            self.conv1 = nn.Conv1d(ninput, 64, 3, padding=1)  
            self.conv2 = nn.Conv1d( 64,  64, 3, padding=1)  
            self.conv3 = nn.Conv1d( 64, 128, 3, padding=1)
            self.conv4 = nn.Conv1d(128, 128, 3, padding=1)
            self.conv5 = nn.Conv1d(128, 128, 3, padding=1)
            self.conv6 = nn.Conv1d(128, 256, 3, padding=1)
            self.conv7 = nn.Conv1d(256, 256, 3, padding=1)
            self.conv8 = nn.Conv1d(256, 512, 3, padding=1)
            self.conv9 = nn.Conv1d(512, 512, 3, padding=1)

            self.pool = nn.MaxPool1d(2, 2)
            self.relu = nn.LeakyReLU(negative_slope=0.5)

            # decoder
            self.t_conv1 = nn.ConvTranspose1d(512, 512, 2, stride=2)
            self.t_conv2 = nn.ConvTranspose1d(512, 256, 2, stride=2)
            self.t_conv3 = nn.ConvTranspose1d(256, 256, 2, stride=2)
            self.t_conv4 = nn.ConvTranspose1d(256, 128, 2, stride=2)
            self.t_conv5 = nn.ConvTranspose1d(128, 128, 2, stride=2)
            self.t_conv6 = nn.ConvTranspose1d(128,  64, 2, stride=2)
            self.t_conv7 = nn.ConvTranspose1d( 64,  64, 2, stride=2)
            self.t_conv8 = nn.ConvTranspose1d( 64, noutput, 2, stride=2)

        elif arch=='nankai_1201':

            #enc = [ 256,  256, 256, 256, 512, 512, 512, 512, 1024]
            #dec = [1024, 1024, 512, 512, 256, 256, 256, 256]

            enc = [ 700, 750, 800, 850, 900, 950, 1000, 1050, 1100]
            dec = [1100,1050,1000, 950, 900, 850, 800, 750]
            #enc = [ 700, 700, 700, 700, 700, 700, 700, 700, 700]
            #dec = [ 700, 700, 700, 700, 700, 700, 700, 700]
            
            self.enc_channels_list = enc
            self.dec_channels_list = dec

            self.conv1 = nn.Conv1d(ninput, enc[0], 7, padding=3)  # 1024
            self.conv2 = nn.Conv1d(enc[0], enc[1], 7, padding=3)  #  512
            self.conv3 = nn.Conv1d(enc[1], enc[2], 7, stride=2, padding=3) # 128
            self.conv4 = nn.Conv1d(enc[2], enc[3], 7, padding=3)  #   64
            self.conv5 = nn.Conv1d(enc[3], enc[4], 7, padding=3)  #   32
            self.conv6 = nn.Conv1d(enc[4], enc[5], 7, padding=3)  #   16
            self.conv7 = nn.Conv1d(enc[5], enc[6], 7, padding=3)  #    8
            self.conv8 = nn.Conv1d(enc[6], enc[7], 7, padding=3)  #    4
            self.conv9 = nn.Conv1d(enc[7], enc[8], 7, padding=3)  #    2

            self.pool = nn.MaxPool1d(2, 2)
            self.relu = nn.LeakyReLU(negative_slope=0.5)

            # decoder
            self.t_conv1 = nn.ConvTranspose1d(dec[0], dec[1], 2, stride=2)    #    2
            self.t_conv2 = nn.ConvTranspose1d(dec[1], dec[2], 2, stride=2)    #    4
            self.t_conv3 = nn.ConvTranspose1d(dec[2], dec[3], 2, stride=2)    #    8
            self.t_conv4 = nn.ConvTranspose1d(dec[3], dec[4], 5, stride=5)    #   16
            self.t_conv5 = nn.ConvTranspose1d(dec[4], dec[5], 3, stride=3)    #   80
            self.t_conv6 = nn.ConvTranspose1d(dec[5], dec[6], 3, stride=3)    #  240
            self.t_conv7 = nn.ConvTranspose1d(dec[6], dec[7], 2, stride=2)    #  720
            self.t_conv8 = nn.ConvTranspose1d(dec[7], noutput,2, stride=2)# 1440

        if set_dropout==True:
            self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        set_dropout = self.set_dropout

        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        if set_dropout==True:
            x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        if set_dropout==True:
            x = self.dropout(x)
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = self.relu(self.conv6(x))
        x = self.pool(x)
        if set_dropout==True:
            x = self.dropout(x)
        x = self.relu(self.conv7(x))
        x = self.pool(x)
        x = self.relu(self.conv8(x))
        x = self.pool(x)
        x = self.relu(self.conv9(x))
        x = self.pool(x)
        if set_dropout==True:
            x = self.dropout(x)

        x = self.relu(self.t_conv1(x))
        x = self.relu(self.t_conv2(x))
        if set_dropout==True:
            x = self.dropout(x)
        x = self.relu(self.t_conv3(x))
        x = self.relu(self.t_conv4(x))
        if set_dropout==True:
            x = self.dropout(x)
        x = self.relu(self.t_conv5(x))
        x = self.relu(self.t_conv6(x))
        if set_dropout==True:
            x = self.dropout(x)
        x = self.relu(self.t_conv7(x))
        x = self.relu(self.t_conv8(x))
        return x

    def encoder(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        x = self.relu(self.conv5(x))
        x = self.pool(x)
        x = self.relu(self.conv6(x))
        x = self.pool(x)
        x = self.relu(self.conv7(x))
        x = self.pool(x)
        x = self.relu(self.conv8(x))
        x = self.pool(x)
        x = self.relu(self.conv9(x))
        x = self.pool(x)
        return x

    def decoder(self, x):
        x = self.relu(self.t_conv1(x))
        x = self.relu(self.t_conv2(x))
        x = self.relu(self.t_conv3(x))
        x = self.relu(self.t_conv4(x))
        x = self.relu(self.t_conv5(x))
        x = self.relu(self.t_conv6(x))
        x = self.relu(self.t_conv7(x))
        x = self.relu(self.t_conv8(x))
        return x


class GNSS_gauge_model():

    def __init__(self,
                 data_path='../data/_sjdf',
                 data_name='sjdf',
                 model_name='sjdf',
                 ndata=959,
                 gnss_stations_list=\
                          ['P316', 'albh', 'bamf', 'bend', 'bils',
                           'cabl', 'chzz', 'cski', 'ddsn', 'eliz', 'elsr',
                           'grmd', 'holb', 'lsig', 'lwck', 'mkah', 'neah',
                           'nint', 'ntka', 'ocen', 'onab', 'oylr', 'p154',
                           'p156', 'p157', 'p160', 'p162', 'p329', 'p343',
                           'p362', 'p364', 'p365', 'p366', 'p380', 'p387',
                           'p395', 'p396', 'p397', 'p398', 'p401', 'p403',
                           'p407', 'p435', 'p441', 'p733', 'p734', 'pabh',
                           'ptrf', 'ptsg', 'reed', 'sc02', 'sc03', 'seas',
                           'seat', 'tfno', 'thun', 'till', 'trnd', 'uclu',
                           'ufda', 'wdcb', 'ybhb'],
                 ngauges=3,
                 npts_in=512,
                 npts_out=256,
                 train_device=None,
                 test_device=None):

        self.data_path = data_path

        if not os.path.exists('_output'):
            os.mkdir('_output')
        self.output_path = '_output'
        
        self.data_fname = None
        self.data_name = data_name

        # data shapes
        self.ndata = ndata
        self.gnss_stations_list = gnss_stations_list
        self.gnss_stations_onoff_array = None
        self.ngnss = len(gnss_stations_list)
        self.ngauges = ngauges # TODO: supply gauges list, then compute length
        self.npts_in = npts_in
        self.npts_out = npts_out

        self.shuffled = None
        self.shuffle_seed = 0
        self.init_weight_seed = 10

        self.model_name = model_name
        self.model_ninput = None
        self.model_noutput = None

        if train_device == None:
            self.train_device = 'cpu'
        elif (train_device == 'cpu') or (train_device == 'cuda'):
            self.train_device = train_device
        else:
            raise ValueError

        if test_device == None:
            self.test_device = 'cpu'
        elif (test_device == 'cpu') or (test_device == 'cuda'):
            self.test_device = test_device
        else:
            raise ValueError

        self.shuffled_batchno = False
        self.data_batches = None

        self.use_Agg = True

        # set dpi for plt.savefig
        self._dpi = 300


    def _batch_data(self, data_array, batch_size):
        """
        Helper function to slice up data array in to batches and return a list
        of arrays

        Parameters
        ----------
        data_array : array_like
            array containing data. Each row (i.e data_array[0, ...]) 
            is assumed to be a data point

        batch_size : int
            max array size along axis=0 for each batch

        Returns
        -------
        batch_list : list
            a list of arrays, each array corresp to a batch

        """

        batch_list = []
        ndata = data_array.shape[0]
        for i in np.arange(0, ndata, batch_size):
            data_pt = data_array[i:(i + batch_size), :, :]
            batch_list.append(data_pt)

        return batch_list


    def _load_array(self, fname, array_type='numpy_array_bin'):
        """
        Helper function to load saved arrays 

        Parameters
        ----------
        fname : string
            The local filename to load

        array_type : {'numpy_array_bin', 
                      'numpy_array_txt', 
                      'numpy_memmap'}
            The data type stored in the file designated by fname

        """

        data_path = self.data_path

        ffname = os.path.join(data_path, fname)

        if array_type == 'numpy_array_bin':
            out = np.load(ffname)
        elif array_type == 'numpy_array_txt':
            out = np.loadtxt(ffname)
        elif array_type == 'numpy_memmap':
            out = np.memmap(ffname, dtype=np.float, mode='r')
        else:
            raise NotImplementedError

        return out


    def load_data(self, batch_size=20):
        '''
        Load interpolated GNSS data for input and gauge data for output
        into memory as a list of arrays. Each array contains a batch.

        Parameters
        ----------
        batch_size :
            choose batch size

        '''

        self.batch_size = batch_size

        data_name = self.data_name

        batch_data = self._batch_data
        load_array = self._load_array
        get_nbatches = self._get_nbatches

        # TODO: init these parameters elsewhere
        ndata = self.ndata
        ngnss = self.ngnss
        ngauges = self.ngauges
        npts_in = self.npts_in  # TODO: npts_gnss might be better var name
        npts_out = self.npts_out

        # load GNSS data
        data_in_Z = load_array('{:s}_gnss_Z.dat'.format(data_name), 
                               array_type='numpy_memmap')
        data_in_E = load_array('{:s}_gnss_E.dat'.format(data_name), 
                               array_type='numpy_memmap')
        data_in_N = load_array('{:s}_gnss_N.dat'.format(data_name), 
                               array_type='numpy_memmap')

        data_in = np.memmap('_output/{:s}_gnss_ENZ.dat'.format(data_name), 
                            mode='w+',
                            shape=(ndata, 3*ngnss, npts_in),
                            dtype=float)

        data_in[:, 0::3, :] = data_in_E.reshape(ndata, ngnss, npts_in)
        data_in[:, 1::3, :] = data_in_N.reshape(ndata, ngnss, npts_in)
        data_in[:, 2::3, :] = data_in_Z.reshape(ndata, ngnss, npts_in)

        # load gauge data
        data_out = load_array('{:s}_gauge.dat'.format(data_name), 
                              array_type='numpy_memmap')
        data_out = data_out.reshape(ndata, ngauges, npts_out)

        # load shuffled indices
        train_index = load_array('{:s}_train_index.txt'.format(data_name), 
                                 array_type='numpy_array_txt').astype(np.int)
        test_index = load_array('{:s}_test_index.txt'.format(data_name),
                                 array_type='numpy_array_txt').astype(np.int)

        # slice into test and training sets
        # creat a list of batches for training, test sets
        data_batches = {}
        data_batches['train'] = {}
        data_batches['test'] = {}

        data_batches['train']['in'] = \
            batch_data(data_in[train_index, : ,:], batch_size)
        data_batches['test']['in'] = \
            batch_data(data_in[test_index, :, :], batch_size)

        data_batches['train']['out'] = \
            batch_data(data_out[train_index, : , :], batch_size)
        data_batches['test']['out'] = \
            batch_data(data_out[test_index, : , :], batch_size)

        self.data_in = data_in
        self.data_out = data_out
        self.data_batches = data_batches

        self.train_index = train_index
        self.test_index = test_index

        self.nbatches_train = get_nbatches('train')
        self.nbatches_test = get_nbatches('test')


    def _get_nbatches(self, dataset):
        """
        Helper function to calculate the number of batches: checks if the
        nbatches for input and output are the same

        """
        data_batches = self.data_batches
        nbatches_in = len(data_batches[dataset]['in'])
        nbatches_out = len(data_batches[dataset]['out'])
        if nbatches_in != nbatches_out:
            raise AssertionError
        else:
            return nbatches_in

    def _get_ndata(self, dataset_name):
        """
        Helper function to calculate the number of data pts in the dataset

        """

        data_batches = self.data_batches

        ndata_in_list = [batch.shape[0] 
                         for batch in data_batches[dataset_name]['in']]
        ndata_in = sum(ndata_in_list)
        ndata_out_list = [batch.shape[0] 
                          for batch in data_batches[dataset_name]['out']]
        ndata_out = sum(ndata_out_list)

        if ndata_in != ndata_out:
            raise AssertionError
        else:
            return ndata_in

    
    def _get_data_batch_tensor(self, 
                               data_subset, 
                               in_out,
                               k,
                               device='cpu',
                               gnss_stations_partial_list=None):
        """
        Helper function to get a specific batch of the data, cast as torch.tensor

        Parameters
        ----------
        data_subset : str {'train', 'test'}
            select batch type

        in_out : str {'in', 'out'}
            select either input or output 

        k : int
            batch number to use
            
        gnss_stations_partial_list : list
            list of GNSS stations to use in input

        Returns
        -------
        out : torch tensor
            tensor containing data batch

        """

        data_batches = self.data_batches
        ngnss = self.ngnss
        gnss_stations_list = self.gnss_stations_list

        data_pt = data_batches[data_subset][in_out][k]

        if in_out == 'in':
            # create a Boolean array indicating which GNSS station to use
            if (type(self.gnss_stations_onoff_array)==type(None)):
                # set self.gnss_stations_onoff_array if not already set
                if gnss_stations_partial_list==None:
                    # raise error if no info provided
                    raise NotImplementedError
                else:
                    # override with provided array
                    gnss_stations_onoff_array = np.zeros(3*ngnss, dtype=bool)

                    for i, station in enumerate(gnss_stations_list):
                        if station in gnss_stations_partial_list:
                            gnss_stations_onoff_array[(3*i):(3*i+3)] = True
                self.gnss_stations_onoff_array = gnss_stations_onoff_array
            else:
                gnss_stations_onoff_array = self.gnss_stations_onoff_array
            
            data_pt = data_pt[:, gnss_stations_onoff_array, :]

        out = torch.tensor(data_pt, dtype=torch.float32).to(device)
        return out


    def _get_data1_tensor(self, i, data_subset='test', inout='out'):

        batch_size = self.batch_size

        k = i // batch_size
        j = i - k*batch_size

        batch_k = self._get_data_batch_tensor(data_subset, inout, k)
        return batch_k[j, ...]

    def train_ensemble(self,
                       nensemble=25,
                       torch_loss_func=nn.L1Loss,
                       torch_optimizer=optim.Adam,
                       nepochs=500,
                       save_interval=None,
                       gnss_stations_partial_list=None,
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

        gnss_stations_list : list
            list of GNSS stations to use for input during training

        '''

        # store training hyper-parameters
        self.nensemble = nensemble
        self.nepochs = nepochs
        self.torch_optimizer = torch_optimizer.__name__
        self.torch_loss_func = torch_loss_func.__name__

        model_name = self.model_name
        get_data_batch_tensor = self._get_data_batch_tensor

        # select hardware
        device = self.train_device

        # set save interval: save model every ``save_interval`` epochs
        if save_interval == None:
            save_interval = int(nepochs/10)    

        # set random seed
        init_weight_seed = self.init_weight_seed
        torch.random.manual_seed(init_weight_seed)

        # set output path
        output_path = self.output_path

        # load data
        if gnss_stations_partial_list == None:
            ngnss_in = self.ngnss
            model_ninput = 3*ngnss_in
        else:
            ngnss_in = len(gnss_stations_partial_list)
            model_ninput = 3*ngnss_in

        model_noutput = self.ngauges
        nbatches_train = self.nbatches_train

        self.ngnss_in = ngnss_in
        self.model_ninput = model_ninput
        self.model_noutput = model_noutput
        self.train_lr = lr

        npts_in = self.npts_in
        npts_out = self.npts_out

        # define a dummy model and save info
        # TODO: set the enc / dec channels list here and pass it to CNN
        model = CNN(model_ninput, model_noutput, arch=model_name)
        model.to(device)

        self.enc_channels_list = model.enc_channels_list
        self.dec_channels_list = model.dec_channels_list
        self.save_model_info()              # save model info

        for n_model in range(nensemble):

            # define new model
            model = CNN(model_ninput, model_noutput, arch=model_name)
            model.to(device)

            self.enc_channels_list = model.enc_channels_list
            self.dec_channels_list = model.dec_channels_list

            # train model
            loss_func = torch_loss_func()
            optimizer = torch_optimizer(model.parameters(), 
                                        lr=lr, 
                                        weight_decay=weight_decay)

            # epochs
            train_loss_array = np.zeros(nepochs)

            for epoch in range(1, nepochs+1):
                # monitor training loss
                train_loss = 0.0

                # train model over each batch
                for k in range(nbatches_train):
                    data_in = get_data_batch_tensor('train', 'in', k,       
                                                    device=device, 
                                                    gnss_stations_partial_list=gnss_stations_partial_list)
                    data_out = get_data_batch_tensor('train', 'out', k,       
                                                     device=device)
                    optimizer.zero_grad()

                    model_out = model(data_in)
                    loss = loss_func(model_out, data_out)

                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                        
                # avg training loss per epoch
                avg_train_loss = train_loss/nbatches_train
                train_loss_array[epoch-1] = avg_train_loss 

                # display status
                msg = '\n ensemble no = {:4d}, epoch = {:4d}, loss = {:1.8f}'\
                     .format(n_model,epoch,avg_train_loss)
                sys.stdout.write(msg)
                sys.stdout.flush()

                fname = '{:s}_gnss_train_loss_m{:02d}.npy'.format(model_name, n_model)
                save_fname = os.path.join(output_path, fname)
                np.save(save_fname, train_loss_array[:epoch])

                if ((epoch) % save_interval) == 0:
                    # save intermediate model
                    fname ='{:s}_gnss_model_m{:04d}_e{:04d}.pkl'\
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
                     self.nepochs,
                     self.nensemble,
                     self.ngauges,
                     self.gnss_stations_list,
                     self.gnss_stations_onoff_array,
                     self.ngnss,
                     self.ngnss_in,
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
                     self.train_device,
                     self.test_device]
        
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
         self.nepochs,
         self.nensemble,
         self.ngauges,
         self.gnss_stations_list,
         self.gnss_stations_onoff_array,
         self.ngnss,
         self.ngnss_in,
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
         self.train_device,
         self.test_device] = info_dict

        if device != None:
            self.device = device

        # load data
        self.load_data(batch_size=self.batch_size)


    def predict_dataset(self, 
                        epoch, 
                        model_list=None,
                        data_subset_list=['train', 'test'], 
                        gnss_stations_partial_list=None,
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

        model_name = self.model_name

        # load data and data dimensions
        if self.data_batches == None:
            self.load_data()

        # if model_list is not provided, predict for all models in ensemble
        if model_list == None:
            model_list = [i for i in range(self.nensemble)]

        get_data_batch_tensor = self._get_data_batch_tensor
        get_ndata = self._get_ndata
        get_nbatches = self._get_nbatches
        get_eval_model = self.get_eval_model

        # TODO: store this somewhere?
        npts_in = self.npts_in 
        npts_out = self.npts_out
        ngauges = self.ngauges

        for data_subset in data_subset_list:
            for n in model_list:

                # large numpy array for all predictions on the dataset
                ndata = get_ndata(data_subset)

                fname = '{:s}_gnss_pred_{:s}-input_m{:04d}_e{:04d}.dat'\
                        .format(model_name, data_subset, n, epoch)
                save_fname = os.path.join('_output', fname)

                pred_all = np.memmap(save_fname,
                                     mode='w+',
                                     dtype=float,
                                     shape=(ndata, ngauges, npts_out))

                model = get_eval_model(n, epoch, device=device)

                nbatches = get_nbatches(data_subset)
                kb = 0
                for i in range(nbatches): 

                    msg = '\rmodel={:6d}, batch={:6d}'.format(n, i)
                    sys.stdout.write(msg)
                    sys.stdout.flush()

                    # setup input data
                    data_in = get_data_batch_tensor(data_subset, 'in', i,      
                        gnss_stations_partial_list=gnss_stations_partial_list)
                    batch_size = data_in.shape[0]

                    # evaluate model
                    with torch.no_grad():
                        model_out = model(data_in)

                    # collect predictions
                    ke = kb + batch_size
                    pred_all[kb:ke, :, :] = model_out.detach().numpy()

                    kb = ke


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
        fname = '{:s}_gnss_model_m{:04d}_e{:04d}.pkl'\
                .format(model_name, n, epoch)

        load_fname = os.path.join('_output', fname)
        model = torch.load(load_fname,
                           map_location=torch.device(device))

        model.eval()

        return model


    def predict_input(self, model_input, epoch, device='cpu'):
        r"""
        Predict all of the data set, both training and test sets

        Parameters
        ----------

        model_input : tensor
            model input of size (?, ninput_gauges, npts)

        epoch : int
            use model after training specified number of epochs

        device : {'cpu', 'cuda'}, default 'cpu'
            choose device for PyTorch modules

        Returns
        -------
        
        pred : list of arrays
            prediction results for each prediction time T, each item is a numpy
        array of the shape (nensemble, ?, ngauges, npts)

        Notes
        -----

        items in output pred is also saved to output directory in binary 
        numpy array format

        """

        # load data and data dimensions
        batch_size = self.batch_size

        nensemble = self.nensemble
        top = self.top

        gauges = np.array(self.gauges)      
        ngauges = self.ngauges
        input_gauges = self.input_gauges
        input_gauges_bool = self.input_gauges_bool

        ndata = model_input.shape[0]
        npts = model_input.shape[-1]

        model_input = torch.tensor(model_input, dtype=torch.float32)
        t_unif = np.linspace(0.0, 4.0, npts)    # TODO: store this elsewhere?

        device = self.device

        model_name = self.model_name
        if epoch == 0:
            epoch = self.nepochs     # use final epoch
        
        pred_all = []
        # predict dataset
        for T in top:
            pred = np.zeros((nensemble, ndata, ngauges, npts))

            for n in range(nensemble):

                model = self.get_eval_model(T, n, epoch, device=device)

                data0 = model_input.to(device)

                # setup input data
                datak = data0[:, input_gauges_bool, :].detach().clone()
                datak[:, :, T:] = 0.0
                datak = datak

                # evaluate model
                with torch.no_grad():
                    model_out = model(datak)
                model_out = model_out.detach().numpy()

                # collect predictions
                pred[n, ...] = model_out

            # save output to _output
            fname = '{:s}_{:s}_{:02d}_{:02d}_{:04d}.dat'\
                     .format(model_name, 'input', T, n, epoch)
            save_fname = os.path.join('_output', fname)
            np.save(save_fname, pred)
            
            pred_all.append(pred)
        
        return pred_all

    def _get_pred(self, m, e, data_subset='test'):

        get_ndata = self._get_ndata

        model_name = self.model_name

        ndata = get_ndata(data_subset)
        ngauges = self.ngauges
        npts_out = self.npts_out

        fname = '{:s}_gnss_pred_{:s}-input_m{:04d}_e{:04d}.dat'\
                .format(model_name, data_subset, m, e)
        save_fname = os.path.join('_output', fname)

        pred = np.memmap(save_fname,
                         mode='r+',
                         dtype=float,
                         shape=(ndata, ngauges, npts_out))
        return pred

    def make_pred_plot(self, 
                       data_subset_list=['test', 'train'],
                       model_no_list=[0],
                       epochs_list=[300],
                       gauge_labels=None,
                       tf_gauge=6.0,
                       tf_gnss=512/3600):
        r'''
        Make prediction plots (run pred_dataset first)

        '''

        import matplotlib.pyplot as plt

        ngauges = self.ngauges

        model_name = self.model_name
        npts_in = self.npts_in
        npts_out = self.npts_out
        get_pred = self._get_pred
        get_ndata = self._get_ndata
        get_data1_tensor = self._get_data1_tensor

        t1 = np.linspace(0.0,tf_gnss,npts_in+1)
        tgnss = 0.5*(t1[1:] + t1[:-1])

        t0 = np.linspace(0.0,tf_gauge,npts_out+1)
        tgauge = 0.5*(t0[1:] + t0[:-1])

        if not os.path.exists('_plots'):
            os.mkdir('_plots')

        if type(gauge_labels) == type(None):
            gauge_labels = ['{:d}'.format(i) for i in range(ngauges)]

        # TODO: make loss plots

        for data_subset in data_subset_list:

            for e in epochs_list:
                ndata = get_ndata(data_subset)

                etamax_pred = np.zeros((ndata, ngauges))
                etamax_obs = np.zeros((ndata, ngauges))

                plot_path = '_plots/e{:04d}'.format(e)
                if not os.path.exists(plot_path):
                    os.mkdir(plot_path)

                for i in range(ndata):

                    obs = get_data1_tensor(i, 
                                           data_subset=data_subset, inout='out')
                    obs = obs.detach().numpy()
                    pred_all = []
                    for m in model_no_list:
                        pred = get_pred(m, e, data_subset=data_subset)
                        pred_all.append(pred)
                    
                    pred_all = np.array(pred_all)
                    
                    fig, axes = plt.subplots(figsize=(10, 3*ngauges), 
                                             nrows=ngauges)
                    for gi in range(ngauges):
                        ax = axes[gi]
                        ax.cla()
                        mean_pred = np.mean(pred_all[:, i, gi, :], axis=0)
                        std_pred = np.std(pred_all[:, i, gi, :], axis=0)

                        etamax_pred[i, gi] = mean_pred.max()
                        etamax_obs[i, gi] = obs[gi, :].max()

                        #ax.plot(tgnss, gnss[i, :, :].T,alpha=0.5)

                        title='{:s} {:4d}'.format(data_subset, i)
                        ax.set_title(title)
                        ax.fill_between(tgauge, 
                                        mean_pred - 2.0*std_pred, 
                                        mean_pred + 2.0*std_pred, 
                                        color='b', 
                                        label='pred $\pm$ 2std', 
                                        alpha=0.2)
                        ax.plot(tgauge, 
                                obs[gi, :], 
                                'k--', 
                                label='obs {:d}'.format(gi))
                        ax.plot(tgauge, mean_pred, 'b', 
                                label='pred {:d}'.format(gi))
                        ax.set_xlabel('hours since earthquake initiation')
                        ax.set_ylabel('$\eta$ (m)')
                        ax.legend()
                    fig.tight_layout()


                    fname = '{:s}_pred_{:s}_r{:04d}_e{:04d}.png'\
                        .format(model_name, data_subset, i, e)
                    ffname = os.path.join(plot_path, fname)
                    fig.savefig(ffname, dpi=300)
                    plt.close(fig)

                    sys.stdout.write('\r {:s}'.format(fname))
                    sys.stdout.flush()

                for gi in range(ngauges):
                    fig_pvo, ax_pvo = plt.subplots(figsize=(5,4))

                    ax_pvo.plot(etamax_obs[:, gi], 
                                etamax_pred[:, gi], 
                                'b.', markersize=4)
                    vmax = max(max(etamax_pred[:, gi]),
                               max(etamax_obs[:, gi]))*1.05

                    ax_pvo.set_aspect('equal')
                    ax_pvo.plot([0, vmax], [0, vmax], 'k--')
                    ax_pvo.set_ylabel('pred')
                    ax_pvo.set_xlabel('obs')
                    ax_pvo.set_xlim([0, vmax])
                    ax_pvo.set_ylim([0, vmax])
                    fname = '{:s}_etamax_{:s}_g{:04d}_e{:04d}.png'.format(model_name, data_subset, gi, e)
                    ffname = os.path.join(plot_path, fname)
                    fig_pvo.savefig(ffname, dpi=300)
                    plt.close(fig_pvo)



                        



            
