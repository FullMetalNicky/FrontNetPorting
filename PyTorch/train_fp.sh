#!/bin/bash
# Francesco Conti <fconti@iis.ee.ethz.ch>
#
# Copyright (C) 2019 ETH Zurich
# All rights reserved
#
# This script runs a full precision training example

CUDA_VISIBLE_DEVICES=1

python3 ETH.py --regime regime.json --epochs 100 --gray 1 --load-trainset "/home/nickyz/data/train_vignette4.pickle" --load-testset "/home/nickyz/data/test_vignette4.pickle" --save-model "Models/DronetGray.pt" 

