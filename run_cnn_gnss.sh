#!/bin/bash

echo "Processing data" \
&& bsub -K < data/cnn_gnss_data.bsub \
&& echo "training model" \
&& bsub -K < scripts/cnn_gnss_train.bsub \
&& echo "testing model" \
&& bsub -K < scripts/cnn_gnss_test.bsub \
&& echo "plotting prediction results" \
&& bsub -K < scripts/cnn_gnss_pred_plot.bsub \

