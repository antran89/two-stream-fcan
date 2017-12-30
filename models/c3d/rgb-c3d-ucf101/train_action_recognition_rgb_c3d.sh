#!/bin/bash

INSTALL=../../../lib/my-very-deep-caffe/cmake-build/tools

start=$(date +%s)

$INSTALL/caffe train \
--solver=c3d_rgb_solver.prototxt \
--gpu=0,1,2,3 \
--weights=pre-trained-models/conv3d_deepnetA_sport1m_iter_1900000_newcaffe_format_rgb_prefix.caffemodel
	
echo "Done~!"

end=$(date +%s)

let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds
echo "Experiments finished at $(date)"

exit
