#!/usr/bin/env bash
# Create the rgb lmdb inputs

SCRIPT_NAME="$0"
if [[ $# < 1 || $# > 3 ]]; then
  printf 'Usage: %s dataset_folder resp_flow_database [result_database_folder]\n' $SCRIPT_NAME
  exit
fi
if [ ! -d $1 ]; then
  echo 'First argument must be directory'
  exit
fi
if [[ $# == 3 && ! -d $3 ]]; then
  echo 'Second argument must be directory'
  exit
fi

if [[ $# == 3 ]]; then
  DATA_BASE_FOLDER=$3
else
  DATA_BASE_FOLDER='.'
fi

# arguments
DATASET_FOLDER=$1
SPLIT=1
RESP_FLOW_DATABASE=$2

# some options variable
SHUFFLED_TRAIN_FILE=$DATA_BASE_FOLDER/$(printf "hollywood2_train_rgb_split%02d.txt" $SPLIT)
SHUFFLED_TEST_FILE=$DATA_BASE_FOLDER/$(printf "hollywood2_val_rgb_split%02d.txt" $SPLIT)
TRAIN_KEY_FILE=$RESP_FLOW_DATABASE/train_snippets/train_lmdb_keys.txt
TEST_KEY_FILE=$RESP_FLOW_DATABASE/val_snippets/test_lmdb_keys.txt
PREFIX_PATH='in_r'
IN_FOLDER_LINK=$DATA_BASE_FOLDER/$PREFIX_PATH

if [ -L $IN_FOLDER_LINK ]; then
  rm $IN_FOLDER_LINK
fi
ln -s $DATASET_FOLDER $IN_FOLDER_LINK

# generate shuffled train/test files
python generate_corresponding_shuffled_train_test_list.py --dataset_folder=$PREFIX_PATH \
  --split=$SPLIT --train_key_file=$TRAIN_KEY_FILE --test_key_file=$TEST_KEY_FILE \
  --shuffled_train_list_file=$SHUFFLED_TRAIN_FILE --shuffled_test_list_file=$SHUFFLED_TEST_FILE
