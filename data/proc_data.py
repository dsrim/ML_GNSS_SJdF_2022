#!/user/bin/env python3
r"""
Pre-process GeoClaw gauge data, and interpolate to a uniform grid
of a time-window

Notes
-----

After running this script: to access interpolated train and test sets

    data = np.load('_sjdf/sjdf.npy')
    train_index = np.load('_sjdf/sjdf_train_index.txt')
    test_index = np.load('_sjdf/sjdf_test_index.txt')

    data_train = data[train_index, ...]
    data_test = data[test_index, ...]

"""

import numpy as np
import os, sys

def interp_gcdata(npts=256,
                  nruns=1300,
                  gauge_path_format=\
                  '_sjdf/SJdF_processed_gauge_data/run_{:06d}/gauge{:05d}.txt',
                  training_set_ratio=0.8,
                  gauges=[702, 901, 911],
                  gauge_data_cols=[0, 1],
                  skiprows=2,
                  thresh_vals=[0.1, 0.5],
                  apply_window=True,
                  win_length=4*3600.0,
                  data_path='_sjdf',
                  dataset_name='sjdf',
                  make_plots=False,
                  use_Agg=False):
    '''
    Process GeoClaw gauge output and output time-series interpolated on a
    uniform grid with npts grid pts.


    Parameters
    ----------

    npts : int, default 256
        no of pts on the uniform time-grid

    skiprows : int, default 2
        number of rows to skip reading GeoClaw gauge output

    nruns : int, default 1300
        total number of geoclaw runs

    gauge_path_format : string, default  'run_{:06d}/_output/gauge_{:05d}.txt'
        specify sub-directory format of geoclaw output, for example
            'run_{:06d}/_output/gauge_{:05d}.txt'
        the field values will contain run and gauge numbers

    training_set_ratio : float, default 0.8
        the ratio of total runs to be set as the training set

    gauges : list of ints, default [702, 901, 911]
        specify the gauge numbers

    gauge_data_cols : list of ints, default [1, 5]
        designate which columns to use for time and surface elevation
        in the GeoClaw gauge ouput file

    skiprows : int, default  2
        number of rows to skip when reading in the GeoClaw gauge output file

    thresh_vals : list of floats, [0.1, 0.5]
        designate threshold values to impose on the gauge data:
        excludes all runs with:
        abs(eta) < thresh_vals[0] in the first gauge
        abs(eta) < thresh_vals[1] in the last gauge

    win_length : float, default 4*3600.0,
        length of the time window in seconds

    data_path : str, default '_sjdf'
        output path to save interpolation results and other information

    dataset_name : str, default 'sjdf'
        name the dataset

    make_plots : bool, default False
        set to True to generate individual plots for each of the runs

    use_Agg : bool, default False
        set to True to use 'Agg' backend for matplotlib, relevant only when
        kwarg make_plots is set to True

    '''

    ngauges = len(gauges)

    data_all = np.zeros((nruns, ngauges, npts))
    i_valid = np.zeros(nruns, dtype=bool)
    t_win_all = np.zeros((nruns, 2))

    if use_Agg:
        import matplotlib
        matplotlib.use('Agg')

    if make_plots:
        import matplotlib.pyplot as plt
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(12,4))


    for i,run_no in enumerate(range(nruns)):
        run_data_buffer = []

        # collect gauge data from run
        for k,gauge_no in enumerate(gauges):

            gauge_fname = gauge_path_format.format(run_no,gauge_no)
            data = np.loadtxt(gauge_fname,skiprows=skiprows,delimiter=',')

            sys.stdout.write(
            '\rrun_no = {:6d}, data npts = {:6d}, unif npts = {:d}'.format(
                run_no, data.shape[0], npts))

            # extract relevant columns
            run_data_buffer.append(data[:, gauge_data_cols])

        valid_data, t_win_lim = _get_window(run_data_buffer,
                                            thresh_vals,
                                            win_length)

        i_valid[i] = valid_data
        if valid_data:

            if apply_window:
                t0 = t_win_lim[0]
                t1 = t_win_lim[1]
                t_unif = np.linspace(t0,t1 + 60*60.0, npts)
                t_win_all[run_no, 0] = t0
                t_win_all[run_no, 1] = t1
            else:
                t_unif = np.linspace(0.0, 6.0*60*60, npts)
                t_win_all[run_no, 0] = 0.0
                t_win_all[run_no, 1] = 6.0*60*60

            for k in range(ngauges):
                raw = run_data_buffer[k]
                eta_unif = np.interp(t_unif, raw[:, 0], raw[:, 1])
                data_all[run_no, k, :] = eta_unif

            sys.stdout.write('   --- valid --- ')

        else:
            pass

        if make_plots:
            ax.cla()
            ax.plot(data[:,0],data[:,1])
            ax.plot(t_unif,eta_unif)
            fig_title = \
                "run_no {:06d}, gauge {:05d}".format(run_no, gauge_no)
            ax.set_title(fig_title)

            fig_fname = \
                '_plots/run_{:06d}_gauge{:05d}.png'.format(run_no,gauge_no)
            fig.savefig(fig_fname, dpi=200)

    data_all = data_all[i_valid, :, :]
    t_win_all = t_win_all[i_valid, :]

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



def _get_window(run_data, thresh_vals, win_length):
    r"""
    Check if data satisfies the requirements: threshold eta at 702

    Parameters
    ----------
    run_data :
        list containing unstructure time reading from the three gauges

    thresh_vals :
        array of size 2 with thresholds for 702, 901

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
    flag = np.zeros(2, dtype=bool)

    t_win_lim = np.zeros(2)

    # apply threshold to 702
    gaugei_data = run_data[0]

    t   = gaugei_data[:, 0]
    eta = gaugei_data[:, 1]

    t_init = t[0]
    t_final = t[-1]

    i_thresh = (np.abs(eta) >= thresh_vals[0])

    if (np.sum(i_thresh) > 0):
        t_win_init = np.min(t[i_thresh])
        t_win_final = t_win_init + win_length

        t_win_lim[0] = t_win_init
        t_win_lim[1] = t_win_final

        if t_win_final <= t_final:
            flag[0] = True

    # apply threshold to 901
    gaugei_data = run_data[1]

    t   = gaugei_data[:, 0]
    eta = gaugei_data[:, 1]

    i_thresh = (np.abs(eta) >= thresh_vals[1])

    if (np.sum(i_thresh) > 0):
        flag[1] = True

    # are both tresholds satisfied?
    valid_data = np.all(flag)

    return valid_data, t_win_lim


def shuffle_dataset(dataset_name='sjdf', data_path='_sjdf', seed=10000):
    """
    Shuffle interpolated dataset. Run after interp_gcdata()

    Parameters
    ----------

    dataset_name : str, default 'sjdf'
        set dataset_name, the function requires the file
            '{:s}/{:s}_runno.npy'.format(data_path, dataset_name)

    data_path : str, default '_sjdf'
        output path to save interpolation results and other information

    seed : int, default 12345
        Random seed supplied to np.random.shuffle()

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

    train_index = int(0.8*ndata)

    # filenames to save
    output_list = [('{:s}_train_runno.txt'.format(dataset_name),
                    shuffled_gc_runno[:train_index]),
                   ('{:s}_train_index.txt'.format(dataset_name),
                    shuffled_indices[:train_index]),
                   ('{:s}_test_runno.txt'.format(dataset_name),
                    shuffled_gc_runno[train_index:]),
                   ('{:s}_test_index.txt'.format(dataset_name),
                    shuffled_indices[train_index:])]

    for fname, array in output_list:
        full_fname = os.path.join(data_path,fname)
        np.savetxt(full_fname, array, fmt='%d')


def convert_to_memmap(data_path='_sjdf',
                  dataset_name='sjdf'):

    fname = os.path.join(data_path, '{:s}.npy'.format(dataset_name))
    data_array = np.load(fname)

    fname = os.path.join(data_path, '{:s}_gauge.dat'.format(dataset_name))
    data_memmap = np.memmap(fname, dtype=float, mode='w+', shape=data_array.shape)
    data_memmap[...] = data_array[...]

    data_memmap.flush()


def convert_to_memmap_gnss(data_path='_sjdf',
                           station_list=['P316', 'albh', 'bamf', 'bend', 'bils',
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
                           orientation_list=['E','N','Z'],
                           npts=512,
                           ndata=959):
    """
    Convert GNSS data into numpy memmap format.
    Before running, the numpy binary array version of the data must be put in data_path, in the format

        '{:s}/gnss_{:s}_{:s}.npy'.format(data_path, station, orientation)

    where station is one of the 'station_list' in the default kwarg, and
    orientation is one of 'orientation_list' in the default kwarg.
    """

    ngnss = len(station_list)

    for orientation in orientation_list:

        fname = os.path.join(data_path,
                             'sjdf_gnss_{:s}.dat'.format(orientation))

        data_input = np.memmap(fname,
                                dtype='float',
                                mode='w+',
                                shape=(ndata, ngnss, npts))

        for i in range(ngnss):
            station_name = station_list[i]
            fname = os.path.join(data_path, 'gnss_{:s}_{:s}.npy'.format(station_name, orientation))
            gnss_station = np.load(fname)
            data_input[:, i, :] = gnss_station.T

        data_input.flush()


if __name__ == "__main__":

    ## threshold, interpolate on window, save data
    interp_gcdata(apply_window=False)

    ## shuffle run numbers, separate training vs test sets, store data
    shuffle_dataset()

    ## create numpy memmapped version of the data
    convert_to_memmap()

    # create numpy memmapped version of the data
    convert_to_memmap_gnss()
