#!/bin/bash
# Francesco Conti <fconti@iis.ee.ethz.ch>
#
# Copyright (C) 2019 ETH Zurich
# All rights reserved
#
# This script runs a full precision training example

CUDA_VISIBLE_DEVICES=1

python3 ETH.py --regime regime.json --epochs 10 --load-trainset "/home/nickyz/data/train.pickle" --load-testset "/home/nickyz/data/test.pickle" --load-model "Models/FrontNet-097.pkl" --quantiz

