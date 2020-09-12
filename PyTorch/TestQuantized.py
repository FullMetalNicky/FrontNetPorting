from __future__ import print_function

import numpy as np

from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from Dataset import Dataset
from torch.utils import data
from ModelManager import ModelManager
import torch

from ConvBlock import ConvBlock
from PenguiNet import PenguiNet
import nemo
import cv2
import os

import argparse
import json

import logging
import platform

def Parse(parser):

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    # [NeMO] Model saving/loading improved for convenience
    parser.add_argument('--save-model', default=None, type=str,
                        help='for saving the model')
    parser.add_argument('--load-model', default=None, type=str,
                        help='for loading the model')

    parser.add_argument('--load-trainset', default=None, type=str,
                        help='for loading the train dataset')

    parser.add_argument('--load-testset', default=None, type=str,
                        help='for loading the test dataset')

    # [NeMO] If `quantize` is False, the script operates like the original PyTorch example
    parser.add_argument('--quantize', default=False, action="store_true",
                        help='for loading the model')

    # [NeMO] If `TrainQ` is True, finetune/relax
    parser.add_argument('--trainq', default=False, action="store_true",
                        help='for loading the model')
    # [NeMO] The training regime (in JSON) used to store all NeMO configuration.
    parser.add_argument('--regime', default=None, type=str,
                        help='for loading the model')
    parser.add_argument('--gray', default=None, type=int,
                        help='for choosing the model')
    args = parser.parse_args()

    return args


def LoadData(args):
    size = "160x96"
    DATA_PATH = "/Users/usi/PycharmProjects/data/" + size + "/"
    name = size + "PaperTestsetPrune2.pickle"
    [x_test, y_test,] = DataProcessor.ProcessTestData(DATA_PATH + name)

    test_set = Dataset(x_test, y_test)
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 1}
    test_generator = data.DataLoader(test_set, **params)

    return test_generator


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch FrontNet')
    args = Parse(parser)

    torch.manual_seed(args.seed)

    # [NeMO] Setup of console logging.
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename="log.txt",
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


    test_generator = LoadData(None)

    #[NeMO] Loading of the JSON regime file.
    regime = {}
    if args.regime is None:
        print("ERROR!!! Missing regime JSON.")
        raise Exception
    else:
        with open(args.regime, "r") as f:
            rr = json.load(f)
        for k in rr.keys():
            try:
                regime[int(k)] = rr[k]
            except ValueError:
                regime[k] = rr[k]

    h = 96
    w = 160
    c = 16

    model = PenguiNet(ConvBlock, [1, 1, 1], True, h=h, w=w, c=c, fc_nodes=960)

    model = nemo.transform.quantize_pact(model, dummy_input=torch.ones((1, 1, h, w)).to("cpu"))
    logging.info("[ETHQ] Model: %s", model)
    epoch, prec_dict = ModelManager.ReadQ("Models/Q/PenguiNetfinal160x96x16.pth", model)
    trainer = ModelTrainer(model, args, regime)
    trainer.Test(test_generator)



if __name__ == '__main__':
    main()
