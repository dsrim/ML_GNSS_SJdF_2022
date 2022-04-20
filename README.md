# ML GNSS SJdF 2022
Code repository to accompany the manuscript:

**Tsunami Early Warning from Global Navigation Satellite System
Data using Convolutional Neural Networks**
<br>
by D. Rim, R. Baraldi, C. M. Liu, R. J. LeVeque, and K. Terada

## Usage

1. Download the processed version of the GNSS dataset from
https://wustl.box.com/v/sjdf-gnss-data into ``data/_sjdf``

2. Execute scripts
   1. If the user's system supports running LSF job scripts, execute the script ``run_cnn_gnss.sh``
   2. Otherwise, manually run commands designated in the LSF scripts using the specified docker image publicly available on [Docker Hub](https://hub.docker.com).
      ```
      data/cnn_gnss_rundata.bsub
      scripts/cnn_gnss_train.bsub
      scripts/cnn_gnss_test.bsub
      scripts/cnn_gnss_pred_plot.bsub
      ```