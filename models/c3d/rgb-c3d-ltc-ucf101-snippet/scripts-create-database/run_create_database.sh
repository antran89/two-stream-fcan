#!/usr/bin/env bash
# Create the rgb flow txt inputs

IMG_FOLDER=/home/tranlaman/Public/data/video/ucf101_tvl1_128_171/img_folder/
RESP_FLOW_DATABASE=/media/tranlaman/data/video-snippets-database/ucf101_tvl1_overlapping_len16_train_test_split1/
LENGTH=16
SPLIT=1
DATABASE_FOLDER=..

bash create_ucf101_corresponding_rgb_snippet.sh $IMG_FOLDER $RESP_FLOW_DATABASE $LENGTH $SPLIT $DATABASE_FOLDER