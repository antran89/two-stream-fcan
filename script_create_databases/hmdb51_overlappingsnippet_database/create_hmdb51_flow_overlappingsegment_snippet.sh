#!/usr/bin/env bash
# Create the flow lmdb inputs

SCRIPT_NAME="$0"
if [ $# != 3 ]; then
  printf 'Usage: %s dataset_folder result_database_folder temporal_length\n' $SCRIPT_NAME
  exit
fi

if [ ! -d $1 ]; then
  echo 'Arguments must be directory'
  exit
fi

DATASET_FOLDER=$1
DATABASE_FOLDER=$2
NEW_LENGTH=$3
SPLIT=1

# some options variable
TIME_STAMP=$(date +%d.%H%M%S)
GRAY=true
IS_FLOW=true
PRESERVE_TEMPORAL=true
SHUFFLED_TRAIN_FILE=$(printf "shuffled_train_list_frm_split%02d_%s.txt" $SPLIT $TIME_STAMP)
SHUFFLED_TEST_FILE=$(printf "shuffled_test_list_frm_split%02d_%s.txt" $SPLIT $TIME_STAMP)
TRAIN_DATABASE='train_snippets'
VAL_DATABASE='val_snippets'
IN_FOLDER_LINK=$(printf 'in_%s' $TIME_STAMP)

if [ -L $IN_FOLDER_LINK ]; then
  rm $IN_FOLDER_LINK
fi
ln -s $DATASET_FOLDER $IN_FOLDER_LINK

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=128
  RESIZE_WIDTH=171
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

# generate shuffled train/test files
python generate_shuffled_flow_overlappingsegment_train_test_list.py --dataset_folder=$IN_FOLDER_LINK \
  --split=$SPLIT --new_length=$NEW_LENGTH \
  --shuffled_train_list_file=$SHUFFLED_TRAIN_FILE --shuffled_test_list_file=$SHUFFLED_TEST_FILE

if [ ! -d $DATABASE_FOLDER/$TRAIN_DATABASE ]; then
	mkdir -p $DATABASE_FOLDER/$TRAIN_DATABASE
fi
if [ ! -d $DATABASE_FOLDER/$VAL_DATABASE ]; then
	mkdir -p $DATABASE_FOLDER/$VAL_DATABASE
fi

mv train_lmdb_keys.txt $DATABASE_FOLDER/$TRAIN_DATABASE
mv test_lmdb_keys.txt $DATABASE_FOLDER/$VAL_DATABASE
rm $SHUFFLED_TRAIN_FILE $SHUFFLED_TEST_FILE $IN_FOLDER_LINK
