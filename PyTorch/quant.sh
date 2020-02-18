#!/bin/bash
# Francesco Conti <fconti@iis.ee.ethz.ch>
#
# Copyright (C) 2019 ETH Zurich
# All rights reserved
#
# This script runs a full precision training example

CUDA_VISIBLE_DEVICES= python3 ETH.py --regime regime.json --epochs 10 --gray 1 --load-trainset "/scratch/fconti/frontnet/data/train_vignette4.pickle" --load-testset "/scratch/fconti/frontnet/data/test_vignette4.pickle" --load-model "Models/DronetGray.pt" --quantiz --save-model "DronetGrayQ.pt"

