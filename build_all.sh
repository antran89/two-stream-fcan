#!/usr/bin/env bash

# # install Caffe dependencies
# sudo apt-get -qq install libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev
# sudo apt-get -qq install --no-install-recommends libboost1.55-all-dev
# sudo apt-get -qq install libgflags-dev libgoogle-glog-dev liblmdb-dev

# # adapt from Yjxiong's code
# # install common dependencies: OpenCV
# # adpated from OpenCV.sh
# version="2.4.13"

# echo "Building OpenCV" $version
# [[ -d 3rd-party ]] || mkdir 3rd-party/
# cd 3rd-party/

# if [ ! -d "opencv-$version" ]; then
#     echo "Installing OpenCV Dependenices"
#     sudo apt-get -qq install libopencv-dev build-essential checkinstall cmake pkg-config yasm libjpeg-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev python-dev python-numpy libtbb-dev libqt4-dev libgtk2.0-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils

#     echo "Downloading OpenCV" $version
#     wget -O OpenCV-$version.zip https://github.com/Itseez/opencv/archive/$version.zip

#     echo "Extracting OpenCV" $version
#     unzip OpenCV-$version.zip
# fi

# echo "Building OpenCV" $version
# cd opencv-$version
# [[ -d build ]] || mkdir build
# cd build
# cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_TBB=ON  -D WITH_V4L=ON ..
# if make -j32 ; then
#     cp lib/cv2.so ../../../
#     echo "OpenCV" $version "built."
# else
#     echo "Failed to build OpenCV. Please check the logs above."
#     exit 1
# fi

# install my-very-deep-caffe
git clone -b rnn-support --single-branch https://github.com/antran89/my-very-deep-caffe.git
cd my-very-deep-caffe
mkdir cmake-build
cd cmake-build
cmake ..

if make -j 10 install ; then
    echo "Caffe Built."
    echo "All tools built. Happy experimenting!"
    cd ../../
else
    echo "Failed to build Caffe. Please check the logs above."
    exit 1
fi


# install compensated dense flow toolbox, 
#alternative, you can install it from Wang Limin's toolbox: https://github.com/yjxiong/dense_flow
git clone -b df_opencv2.4 --single-branch https://github.com/antran89/dense_flow.git
cd dense_flow
mkdir cmake-build
cd cmake-build
cmake ..

if make -j ; then
    echo "Dense Flow built."
else
    echo "Failed to build Dense Flow. Please check the logs above."
    exit 1
fi
