#!/bin/bash

args=("$@")

files=$(gsutil ls gs://sonumator/recordings/2015-2016/*.wav | tail -q -n +${args[0]});

for file in $files; do
    echo "Processing $file"
    python ./sonumator.py classify_file --file $file --training-set ./output/ --output ./csv/
done