#!/bin/bash

CAFFEMODEL=../../c3d_rgb_lmdb_fb_split1_model4_iter_10000.caffemodel

python transfer_c3d_ucf101_model_to_rgb_prefix_format.py $CAFFEMODEL