#!/usr/bin/env bash
# Create the rgb flow txt inputs

# modify IMG_FOLDER path to your dataset
IMG_FOLDER=/home/tranlaman/Public/data/video/Hollywood2_comp_tvl1_128_171/img_folder/
RESP_FLOW_DATABASE=../../../../internal-data/video-snippets-database/hollywood2_tvl1_overlapping_len16_train_test_split1/
DATABASE_FOLDER=..

bash create_hollywood2_corresponding_rgb_snippet.sh $IMG_FOLDER $RESP_FLOW_DATABASE $DATABASE_FOLDER
