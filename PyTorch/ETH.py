from __future__ import print_function
from PreActBlock import PreActBlock
from FrontNet import FrontNet
from Dronet import Dronet
import numpy as np

from DataProcessor import DataProcessor
from ModelTrainerETH import ModelTrainer
from Dataset import Dataset
from torch.utils import data
from ModelManager import ModelManager
import torch
import cv2

import argparse
import json

import logging


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
    # [NeMO] The training regime (in JSON) used to store all NeMO configuration.
    parser.add_argument('--regime', default=None, type=str,
                        help='for loading the model')
    parser.add_argument('--gray', default=None, type=int,
                        help='for choosing the model')
    args = parser.parse_args()

    return args


def LoadData(args):

    if args.gray is not None:
        [x_train, x_validation, y_train, y_validation] = DataProcessor.ProcessTrainData(
            args.load_trainset, 60, 108, True)
        [x_test, y_test] = DataProcessor.ProcessTestData(args.load_testset, 60, 108, True)
    else:
        [x_train, x_validation, y_train, y_validation] = DataProcessor.ProcessTrainData(
            args.load_trainset, 60, 108)
        [x_test, y_test] = DataProcessor.ProcessTestData(args.load_testset, 60, 108)


    training_set = Dataset(x_train, y_train, True)
    validation_set = Dataset(x_validation, y_validation)
    test_set = Dataset(x_test, y_test)

    # Parameters
    num_workers = 6
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': num_workers}
    train_loader = data.DataLoader(training_set, **params)
    validation_loader = data.DataLoader(validation_set, **params)
    params = {'batch_size': args.batch_size,
              'shuffle': False,
              'num_workers': num_workers}
    test_loader = data.DataLoader(test_set, **params)

    return train_loader, validation_loader, test_loader


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

    train_loader, validation_loader, test_loader = LoadData(args)

    # [NeMO] Loading of the JSON regime file.
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

    if args.gray is not None:
        model = Dronet(PreActBlock, [1, 1, 1], True)
    else:
        model = Dronet(PreActBlock, [1, 1, 1], False)

    # [NeMO] This used to preload the model with pretrained weights.
    if args.load_model is not None:
        ModelManager.Read(args.load_model, model)

    trainer = ModelTrainer(model, args, regime)
    if args.quantize:
        #logging.disable(logging.INFO)
        trainer.Quantize(validation_loader)


    #trainer.Train(train_loader, validation_loader)
    #trainer.Predict(test_loader)

    if args.save_model is not None:
        ModelManager.Write(trainer.GetModel(), 100, args.save_model)
        # state_dict = torch.load(args.save_model, map_location='cpu')
        # qmodel = state_dict["model"]
        # for key, value in qmodel.items():
        #     if "bias" in key:
        #         value = value.reshape(-1)
        #         value = list(value.numpy())
        #         logging.info("{}={}".format(key, value))

if __name__ == '__main__':
    main()
