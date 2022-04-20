
import cnn_gnss
import numpy as np

Model = cnn_gnss.GNSS_gauge_model()
Model.load_model('sjdf')

nepochs = Model.nepochs
epochs_list = np.linspace(0, nepochs, 11, dtype=int)[1:]

Model.make_pred_plot(model_no_list=range(25),
                     epochs_list=epochs_list)
