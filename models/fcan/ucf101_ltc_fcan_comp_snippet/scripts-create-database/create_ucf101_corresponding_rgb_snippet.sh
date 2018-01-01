#!/usr/bin/env bash
# Create the rgb lmdb inputs

SCRIPT_NAME="$0"
if [[ $# < 4 || $# > 5 ]]; then
  printf 'Usage: %s dataset_folder resp_flow_database length split [result_database_folder]\n' $SCRIPT_NAME
  exit
fi
if [ ! -d $1 ]; then
  echo 'First argument must be directory'
  exit
fi
if [[ $# == 5 && ! -d $5 ]]; then
  echo 'Fifth argument must be directory'
  exit
fi

if [[ $# == 5 ]]; then
  DATA_BASE_FOLDER=$5
else
  DATA_BASE_FOLDER='.'
fi

# arguments
DATASET_FOLDER=$1
SPLIT=$4
RESP_FLOW_DATABASE=$2
LENGTH=$3

# some options variable
SHUFFLED_TRAIN_FILE=$DATA_BASE_FOLDER/$(printf "ucf101_train_rgb_len%d_split%02d.txt" $LENGTH $SPLIT)
SHUFFLED_TEST_FILE=$DATA_BASE_FOLDER/$(printf "ucf101_val_rgb_len%d_split%02d.txt" $LENGTH $SPLIT)
TRAIN_KEY_FILE=$RESP_FLOW_DATABASE/train_snippets/train_lmdb_keys.txt
TEST_KEY_FILE=$RESP_FLOW_DATABASE/val_snippets/test_lmdb_keys.txt
PREFIX_PATH='in_r'
IN_FOLDER_LINK=$DATA_BASE_FOLDER/$PREFIX_PATH

if [ ! -L $IN_FOLDER_LINK ]; then
  ln -s $DATASET_FOLDER $IN_FOLDER_LINK
fi

# generate shuffled train/test files
python generate_corresponding_shuffled_train_test_list.py --dataset_folder=$PREFIX_PATH \
  --split=$SPLIT --train_key_file=$TRAIN_KEY_FILE --test_key_file=$TEST_KEY_FILE \
  --shuffled_train_list_file=$SHUFFLED_TRAIN_FILE --shuffled_test_list_file=$SHUFFLED_TEST_FILE
