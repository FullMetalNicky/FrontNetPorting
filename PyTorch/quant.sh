#!/bin/bash
# Francesco Conti <fconti@iis.ee.ethz.ch>
#
# Copyright (C) 2019 ETH Zurich
# All rights reserved
#
# This script runs a full precision training example

mkdir -p PenguiNet/golden
CUDA_VISIBLE_DEVICES=2 python3 ETHQuantize.py --regime regime.json --epochs 100 --gray 1 --load-trainset "/home/nickyz/data/160x96HimaxMixedTrain_12_03_20AugCrop.pickle" --load-model "PenguiNetQ.pt" --quantize --save-model "PenguiNetQ2.pt"

