#!/usr/bin/env bash

SCRIPT_NAME="$0"
if [ $# != 3 ]; then
	echo 'The arguments for the program are not correct!'
	printf 'Usage: %s gpu_id caffe_model path_to_out_folder\n' $SCRIPT_NAME
	exit
fi
re='^-?[0-9]+$'
if [[ ! $1 =~ $re ]]; then
	echo 'The first arg should be a gpu id!'
	exit
fi
if [ ! -f $2 ]; then
	echo 'The second arg should be a caffemodel!'
	exit
fi
if [ ! -d $3 ]; then
	echo 'The third arg should be a folder for extracting features!'
	exit
fi

# gpu to run the job (FOR not register memory on GPU 0)
export CUDA_VISIBLE_DEVICES=$1

# some parameters for the job
TOOLS=/home/tranlaman/Desktop/caffe-workspace/my-very-deep-caffe/cmake-build-c3d/tools/
MODEL=$2
GPU_ID=0

# some parameters for the program
MODEL_PROTOTXT=c3d_flow_bn_all_1view_testset.prototxt
BLOB_NAME="flow_prob"
TIME_STAMP=$(date +%d.%H%M%S)
BATCH_SIZE=15
FEATURE_FOLDER=$3
OUT_FOLDER_LINK=$(printf 'out_%s' $TIME_STAMP)

printf 'Extracting features for the model weights: %s\n' $MODEL
printf 'Extracting features for the model prototxt: %s\n' $MODEL_PROTOTXT

# start timing
start=$(date +%s)

if [ -L $OUT_FOLDER_LINK ]; then
	rm $OUT_FOLDER_LINK
fi

# get feature folder name
# filename="${MODEL##*/}"
filename=$(basename "$MODEL")
name="${filename%.*}"
FEATURE_FOLDER=$FEATURE_FOLDER/$name
if [ ! -d $FEATURE_FOLDER ]; then
	mkdir $FEATURE_FOLDER
fi

ln -s $FEATURE_FOLDER $OUT_FOLDER_LINK
python create_ucf101_output_folders.py --output_folder=$OUT_FOLDER_LINK

DATABASE=/home/tranlaman/Public/data/new-caffe-database/ucf101_comp_tvl1_overlapping_segment16_train_test_split1/
TRAIN_LIST_FILE=$(printf 'feature_extraction_train_list_prefix_%s.txt' $TIME_STAMP)
TEST_LIST_FILE=$(printf 'feature_extraction_test_list_prefix_%s.txt' $TIME_STAMP)
TRAIN_KEY_FILE=$DATABASE/train_flow_lmdb/train_lmdb_keys.txt
TEST_KEY_FILE=$DATABASE/val_flow_lmdb/test_lmdb_keys.txt
if [[ ! -f $TRAIN_LIST_FILE || ! -f $TEST_LIST_FILE ]]; then
	python generate_train_test_prefix_file.py --output_folder=$OUT_FOLDER_LINK --batch_size=$BATCH_SIZE \
	--train_key_file=$TRAIN_KEY_FILE \
	--test_key_file=$TEST_KEY_FILE \
	--train_list_file=$TRAIN_LIST_FILE \
	--test_list_file=$TEST_LIST_FILE
fi

# extract features of test data
$TOOLS/extract_1view_features_from_database $MODEL_PROTOTXT \
$MODEL $GPU_ID $BATCH_SIZE 1352000 $TEST_LIST_FILE $BLOB_NAME

wait

rm $OUT_FOLDER_LINK $TRAIN_LIST_FILE $TEST_LIST_FILE

# measuring time
echo "Done~!"
end=$(date +%s)

let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds
echo "Experiments finished at $(date)"

exit