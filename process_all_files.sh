#!/bin/bash

for file in $(gsutil ls gs://sonumator/recordings/2015-2016/*.wav); do
    echo "Processing $file"
    python ./sonumator.py classify_file --file $file --training-set ./output/ --output ./csv/
done