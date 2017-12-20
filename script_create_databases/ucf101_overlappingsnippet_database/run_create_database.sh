#!/usr/bin/env bash
# Create the rgb flow txt inputs

FLOW_FOLDER=/home/tranlaman/Public/data/video/ucf101_comp_tvl1_128_171/flow_folder/
DATASET_FOLDER=/media/tranlaman/data/video-snippets-database/ucf101_tvl1_overlapping_len32_train_test_split1/
TEMPORAL_LENGTH=32
bash create_ucf101_flow_overlappingsegment_snippet.sh $FLOW_FOLDER $DATASET_FOLDER $TEMPORAL_LENGTH