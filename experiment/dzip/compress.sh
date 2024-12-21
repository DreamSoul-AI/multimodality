#!/bin/bash

input_dir="files_to_be_compressed"
output_dir="files_compressed"

# Ensure the output directory exists
mkdir -p "$output_dir"

# Loop through each file in the input directory
for FILE in "$input_dir"/*; do
    # Extract the base filename without the extension
    BASE=${FILE##*/}
    BASE=${BASE%.*}
    JOINT=_
    EXT=bstrap
    # Set the output path with the .dzip extension
    OUTPUT="$output_dir/$BASE.dzip"

    # Run the commands with the current file
    python run.py --file_name "$FILE"
    python compress_adaptive.py --file_name "$BASE" --bs 16 --timesteps 16 --output "$OUTPUT"
done

#FILE=$1
#BASE=${FILE##*/}
#BASE=${BASE%.*}
#JOINT=_
#EXT=bstrap
#OUTPUT=$2
#
#
#python run.py --file_name $FILE
#python compress_adaptive.py --file_name $BASE --bs 32 --timesteps 32 --output $OUTPUT
