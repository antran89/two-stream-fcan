#!/bin/bash

INSTALL=../../../lib/my-very-deep-caffe/cmake-build/tools

start=$(date +%s)

$INSTALL/caffe train \
--solver=c3d_rgb_solver.prototxt \
--gpu=0,1 \
--weights=pre-trained-models/conv3d_deepnetA_sport1m_iter_1900000_newcaffe_format_rgb_prefix.caffemodel,\
pre-trained-models/ucf101_c3d_compensated_flow_bs120_wi1_iter_20000.caffemodel

echo "Done~!"

end=$(date +%s)

let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds
echo "Experiments finished at $(date)"

exit
