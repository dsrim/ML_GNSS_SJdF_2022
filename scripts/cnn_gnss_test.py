
import numpy as np
import os

import cnn_gnss

nstations = 40  # 20 10
fname = '_output/picked_stations_n{:04d}.txt'.format(nstations)
with open(fname, mode='r') as infile:
    gnss_stations_partial_list = infile.readlines()

gnss_stations_partial_list =\
  [s.rstrip('\n') for s in gnss_stations_partial_list]

Model = cnn_gnss.GNSS_gauge_model()
Model.load_data()
Model.load_model('sjdf')

nepochs = Model.nepochs
epochs_list = np.linspace(0, nepochs, 11, dtype=int)[1:]

for epoch in epochs_list:
    Model.predict_dataset(epoch, 
                          model_list=range(25),
                          gnss_stations_partial_list=gnss_stations_partial_list)
