#!/bin/bash
# Francesco Conti <fconti@iis.ee.ethz.ch>
#
# Copyright (C) 2019 ETH Zurich
# All rights reserved
#
# This script runs a full precision training example

CUDA_VISIBLE_DEVICES= python3 ETHQuantize2.py --regime regime.json --epochs 10 --gray 1 --load-trainset "/Users/usi/PycharmProjects/data/160x96/160x96HimaxMixedTrain_12_03_20AugCrop.pickle" --load-model "Models/HannaNet160x96.pt" --trainq --save-model "HannaNetQ.pt"

