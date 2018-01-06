#!/usr/bin/env bash
# Create the rgb flow txt inputs

FLOW_FOLDER=/home/tranlaman/Public/data/video/hmdb51_comp_tvl1_128_171/flow_folder/

DATABASE_FOLDER=../../internal-data/video-snippets-database/hmdb51_tvl1_overlapping_len16_train_test_split1/
TEMPORAL_LENGTH=16
bash create_hmdb51_flow_overlappingsegment_snippet.sh $FLOW_FOLDER $DATABASE_FOLDER $TEMPORAL_LENGTH