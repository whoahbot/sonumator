#!/bin/bash

set -o pipefail

args=("$@")

input_files=$(gsutil ls gs://sonumator/recordings/${args[0]}/*.wav);

trimmed_input_files=("${input_files[@]%.*}")
trimmed_output_files=("${output_files[@]%.*}")

for file in $input_files; do
    base_file_name=$(basename ${file[@]%.*})
    file_exists=$(gsutil ls gs://sonumator/csv/${base_file_name}.csv); rc=$?
    if [ $rc == 1 ]
    then 
        echo "Processing $file"
        time python ./sonumator.py classify_file --file $file --training-set ./output/ --output ./csv/
    fi
done