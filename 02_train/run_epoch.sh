#!/bin/sh

setup="$1"
i=$(ls -1 $setup/epoch*results* | tail -n 1 | egrep -o 'epoch[0-9]*')
i=${i#epoch}
i=$(printf "%02d" $(expr $i + 1))
python3 train.py $setup ../01_data/train-test-split.zarr $setup/model.pth -d | tee $setup/epoch${i}_results_training
python3 train.py $setup ../01_data/train-test-split.zarr $setup/model.pth -d -t test | tee $setup/epoch${i}_results_test
