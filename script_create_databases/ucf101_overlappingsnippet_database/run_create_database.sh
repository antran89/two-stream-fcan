#!/usr/bin/env bash
# Create the rgb flow txt inputs

# modify the path of FLOW_FOLDER to the folder contains compensated optical flows
FLOW_FOLDER=/home/tranlaman/Public/data/video/ucf101_comp_tvl1_128_171/flow_folder/

DATASET_FOLDER=../../internal-data/video-snippets-database/ucf101_tvl1_overlapping_len16_train_test_split1/
TEMPORAL_LENGTH=16
bash create_ucf101_flow_overlappingsegment_snippet.sh $FLOW_FOLDER $DATASET_FOLDER $TEMPORAL_LENGTH