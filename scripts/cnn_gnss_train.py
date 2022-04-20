
import cnn_gnss

nstations = 40  
fname = '_output/picked_stations_n{:04d}'.format(nstations)
with open(fname, mode='r') as infile:
    gnss_stations_partial_list = infile.readlines()

gnss_stations_partial_list =\
  [s.rstrip('\n') for s in gnss_stations_partial_list]


A = cnn_gnss.GNSS_gauge_model()
A.load_data()
A.train_ensemble(nensemble=25, 
                 nepochs=800,
                 gnss_stations_partial_list=gnss_stations_partial_list)
