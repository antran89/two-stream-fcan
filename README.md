# Two-stream Flow-guided Convolutional Attention Networks for Action Recognition

The repository contains different models implemented in [Two-stream FCAN](https://arxiv.org/abs/1708.09268). If you feel this repository useful, please cite our paper:
```
@inproceedings{AnTran_ICCV_2017,
  author    = {An Tran and
               Loong-Fah Cheong},
  title     = {Two-stream Flow-guided Convolutional Attention Networks for Action Recognition},
  booktitle = {The IEEE International Conference on Computer Vision Workshop (ICCVW)},
  year      = {2017},
}
```

## News & Updates

#### Oct 02, 2017

#### Dec 18, 2017

#### Jan 06, 2018

Plan to release the source codes:

- [x] Write build_all.sh script
- [x] Release scripts to create database file
- [x] Release test prototxt file
- [x] Release caffemodel file
- [x] Release classification scripts
- [ ] Write README.md file.

The following is the guidance to reproduce the reported results and extend to more datasets.

-------------
# Usage Guide

## Prerequisites
Similar to [TSN repository](https://github.com/yjxiong/temporal-segment-networks), the major libraries we use are

- [my-very-deep-caffe][caffe]

- [our modifications of dense flow][df].

Our `my-very-deep-caffe` is a modified fork of BLVC [Caffe][BLVC-Caffe] and our `dense_flow` software is modifications from Wang Limin's [dense flow][Wang_dense_flow].

## Software design philosophy
We choose to keep our `my-very-deep-caffe` to be aligned with the original of Caffe's fork commit 5a201dd960840c319cefd9fa9e2a40d2c76ddd73. 
We would like to preserve the strength of BLVC Caffe software which is a deep learning framework made with expression, speed, and modularity 
in mind. Our software also inherits training mechanism from multiple GPUs from BLVC Caffe.

Our models are released under folder `models`. With extensibility and simplicity in mind, we organize steps of training, extracting features and evaluating models of a model in the same folder. If we would like to extend the training to other datasets, we only need to copy a sample folder into a
folder and make some necessary modifications.

## Code & Data Preparation
Use git to clone this repository and its submodules
```
git clone --recursive https://github.com/antran89/two-stream-fcan.git
```

Then run the building scripts to build the libraries.

```
bash build_all.sh
```
It will build Caffe and dense_flow. Since we need OpenCV to have Video IO, which is absent in most default installations, it will also download and build a local installation of OpenCV and use its Python interfaces.

### Get the videos
[[Reference to TSN setup][tsn]]

We experimented on three mainstream action recognition datasets: [UCF-101][ucf101], [HMDB51][hmdb51] and [Hollywood2][hollywood2]. Videos can be downloaded directly from their websites.
After download, please extract the videos from the `rar` archives.
- UCF101: the ucf101 videos are archived in the downloaded file. Please use `unrar x UCF101.rar` to extract the videos.
- HMDB51: the HMDB51 video archive has two-level of packaging.
The following commands illustrate how to extract the videos.
```
mkdir rars && mkdir videos
unrar x hmdb51-org.rar rars/
for a in $(ls rars); do unrar x "rars/${a}" videos/; done;
```

### Extract Frames and Optical Flow Images
To run the training and testing, we need to extract frames of video, also the temporal-C3D networks need optical flow or compensated optical flow images for input.

For UCF101, the extraction can be achieved with the script `lib/dense_flow/python/runme_dense_flow_ucf101.sh`. We can modify some key elements of scripts:
- `VIDEO_FOLDER` points to the folder where you put the video dataset
- `IMG_FOLDER` and `FLOW_FOLDER` point to the root folder where the extracted frames and optical images will be put in
- `NUM_WORKERS` specifies the number of processes to use in parallel for flow extraction on 1 GPU
- `CUDA_VISIBLE_DEVICES` specifies GPU id to run extraction, default gpu 0
- Other variables are self-explainable.

The command for running frames and optical flow extraction is as follows

```
cd lib/dense_flow/python
bash runme_dense_flow_ucf101.sh
```

It will take from several hours to several days to extract optical flows for the whole datasets, depending on the number of GPUs.

### Building file list of video snippets for Caffe training/testing
In order to train/test video classification models, we need to have a text file lists of all video segments. An example of our video snippets database is in `internal-data/video-snippets-database`. We need to generate the video snippets again for each dataset, because data is different for different users (e.g., number of video frames). The format of video snippets as following:
```
PlayingPiano/v_PlayingPiano_g20_c02 0017 63
BlowingCandles/v_BlowingCandles_g14_c03 0016 13
BreastStroke/v_BreastStroke_g21_c02 0064 18
GolfSwing/v_GolfSwing_g14_c03 0011 32
Archery/v_Archery_g16_c05 0046 2
BreastStroke/v_BreastStroke_g21_c01 0073 18
```

with three columns corresponding to video file, frame index, and class id (0-index).

To create database, modify `FLOW_FOLDER` in `run_create_database.sh` file in folder `script_create_databases/ucf101_overlappingsnippet_database`, then run:

```
cd script_create_databases/ucf101_overlappingsnippet_database/
bash run_create_database.sh
```

### Get pretrained models
We provided the trained model weights in Caffe style, consisting of specifications in Protobuf messages, and model weights.
In the codebase we provide the model spec for UCF101 and HMDB51.
The model weights can be downloaded by running the script

```
bash pre-trained-models/get_pretrained_models.sh
```

## Testing Provided Models

## Training Two-stream FCAN Networks

## Contact
For any question, please contact
```
An Tran: an.tran@u.nus.edu
```

[caffe]:https://github.com/antran89/my-very-deep-caffe
[df]:https://github.com/antran89/dense_flow
[BLVC-Caffe]:https://github.com/antran89/dense_flow
[Wang_dense_flow]:https://github.com/yjxiong/dense_flow
[tsn]:https://github.com/yjxiong/temporal-segment-networks
[ucf101]:http://crcv.ucf.edu/data/UCF101.php
[hmdb51]:http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
[hollywood2]:http://www.di.ens.fr/~laptev/actions/hollywood2/
