#!/bin/bash

echo "Processing data" \
&& bsub -K < data/cnn_gnss_data.bsub \
&& echo "training model" \
&& bsub -K < scripts/cnn_gnss_train.bsub \
&& echo "testing model" \
&& bsub -K < scripts/cnn_gnss_test.bsub \
&& echo "plotting prediction results" \
&& bsub -K < scripts/cnn_gnss_pred_plot.bsub \
&& echo "transfering data" \
&& export LSF_DOCKER_ENTRYPOINT=/bin/sh \
&& export LSF_DOCKER_VOLUMES='/home/rim:/home/rim' \
&& bsub -K < scripts/cnn_gnss_transfer.bsub
