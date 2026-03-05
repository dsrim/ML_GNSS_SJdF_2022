
import cnn_gnss
import numpy as np

Model = cnn_gnss.GNSS_gauge_model()
Model.load_model('sjdf')

nepochs = Model.nepochs
epochs_list = [10,]

Model.make_pred_plot(model_no_list=range(3),
                     epochs_list=epochs_list)
