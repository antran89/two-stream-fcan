#!/usr/bin/env bash
# Create the rgb flow txt inputs

SPLIT=1
LENGTH=16
DATABASE_NAME=$(printf "hollywood2_tvl1_overlapping_len%d_train_test_split%d" $LENGTH $SPLIT)
RESP_FLOW_DATABASE=../../../../internal-data/video-snippets-database/$DATABASE_NAME
DATABASE_FOLDER=..

IMG_FOLDER=/home/tranlaman/Public/data/video/Hollywood2_comp_tvl1_128_171/img_folder/
FLOW_FOLDER=/home/tranlaman/Public/data/video/Hollywood2_comp_tvl1_128_171/flow_folder/
bash create_hollywood2_corresponding_flow_snippet.sh $FLOW_FOLDER $RESP_FLOW_DATABASE $LENGTH $DATABASE_FOLDER
bash create_hollywood2_corresponding_rgb_snippet.sh $IMG_FOLDER $RESP_FLOW_DATABASE $LENGTH $DATABASE_FOLDER
