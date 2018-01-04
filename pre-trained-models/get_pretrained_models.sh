#!/usr/bin/env bash

# C3D Sport1M models (adapted to the new version of Caffe in this repo)
# pretrained models for spatial-C3D, temporal-C3D, twostream-FCAN
wget -O conv3d_deepnetA_sport1m_iter_1900000_newcaffe_format.caffemodel --no-check-certificate "https://onedrive.live.com/download?cid=07DB5FA6221DB110&resid=7DB5FA6221DB110%21111&authkey=AMM1nLGJ6_HEUYA"
wget -O conv3d_deepnetA_sport1m_iter_1900000_newcaffe_format_rgb_prefix.caffemodel --no-check-certificate "https://onedrive.live.com/download?cid=07DB5FA6221DB110&resid=7DB5FA6221DB110%21109&authkey=AHV0_i8nQnCe5VM"
wget -O c3d_flow_sport1m_newcaffe_format.caffemodel --no-check-certificate "https://onedrive.live.com/download?cid=07DB5FA6221DB110&resid=7DB5FA6221DB110%21112&authkey=AEVFtuyus5V9Slg"

# spatial C3D models

# temporal C3D models

# FCAN models
wget -O ../models/fcan/ucf101_ltc_fcan_comp_snippet/c3d_rgb_fcan_pool1_sz112_len16_split1_bs64_fi1_iter_20000.caffemodel --no-check-certificate "https://onedrive.live.com/download?cid=07DB5FA6221DB110&resid=7DB5FA6221DB110%21113&authkey=AGAQBULsRAGzGyM"
wget -O ../models/fcan/hmdb51_ltc_fcan_comp_snippet/hmdb51_c3d_rgb_fcan_pool1_sz112_len16_bs64_split1_fi01_iter_10000.caffemodel --no-check-certificate "https://onedrive.live.com/download?cid=07DB5FA6221DB110&resid=7DB5FA6221DB110%21114&authkey=ALzBUW7q9G9saGA"
wget -O ../models/fcan/hollywood2_c3d_fcan_comp_snippet/hollywood2_c3d_rgb_fcan_comp_pool1_sz112_len16_bs64_split1_fi1_iter_10000.caffemodel --no-check-certificate "https://onedrive.live.com/download?cid=07DB5FA6221DB110&resid=7DB5FA6221DB110%21115&authkey=AEo7el4kYoZ098o"
