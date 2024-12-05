#! /bin/bash

# 
data_dir=$1
if [ -z "$data_dir" ]; then
    echo "ERROR: No data directory provided!"
    exit 1
fi
if [ ! -d "$data_dir" ]; then
    mkdir -p $data_dir
cd $data_dir
wget "https://storage.googleapis.com/long-range-arena/lra_release.gz"

