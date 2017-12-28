#!/bin/bash

SCRIPT_NAME="$0"
if (($# < 1)); then
	echo 'The arguments for the program are not correct!'
	printf 'Usage: %s path_to_feature_folder [split [prob_extension]]\n' $SCRIPT_NAME
	exit
fi
if [ ! -d $1 ]; then
	echo 'The first arg should be a folder for extracting features!'
	exit
fi
if (($# >= 2)); then
	re='^[123]$'
	if [[ ! $2 =~ $re ]]; then
		echo 'The second arg should be 1, 2, or 3!'
		exit
	fi
fi

# some variables for the script
FEATURE_FOLDER=$1
if (($# >= 2)); then
	SPLIT=$2
else
	SPLIT=1
fi
if (($# >= 3)); then
	PROB_EXTENSION=$3
else
	PROB_EXTENSION=prob
fi

# folder to contain Python scripts file
INSTALL="$(dirname $(readlink -f "$0"))"

start=$(date +%s)

printf "Starting compute video mAP for Hollywood2 split %s\n" $SPLIT
printf "Feature folder: %s\n" $FEATURE_FOLDER
printf "Probability extention: %s\n" $PROB_EXTENSION

python $INSTALL/classify_video_map_hollywood2.py --feature_folder=$FEATURE_FOLDER --split=$SPLIT --prob_extension=$PROB_EXTENSION
echo "Done~!"
end=$(date +%s)

let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds
echo "Experiments finished at $(date)"

exit
