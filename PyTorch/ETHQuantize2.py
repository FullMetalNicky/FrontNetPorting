from __future__ import print_function

import numpy as np

from DataProcessor import DataProcessor
from ModelTrainerETH import ModelTrainer
from Dataset import Dataset
from torch.utils import data
from ModelManager import ModelManager
import torch

from ConvBlock import ConvBlock
from HannaNet import HannaNet
import nemo
import os

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

    [x_train, x_validation, y_train, y_validation] = DataProcessor.ProcessTrainData(
        args.load_trainset)

    training_set = Dataset(x_train, y_train, True)
    validation_set = Dataset(x_validation, y_validation)

    # Parameters
    # num_workers - 0 for debug in Mac+PyCharm, 6 for everything else
    num_workers = 0
    params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': num_workers}
    train_loader = data.DataLoader(training_set, **params)
    validation_loader = data.DataLoader(validation_set, **params)

    return train_loader, validation_loader


def ExportONXX(model, model_inner, val_loader, validate, h, w):
    # print(model)
    # model_inner = model
    # print("before export")
    # model.graph.print_jit_graph()
    nemo.utils.export_onnx("HannaNet/model_int.onnx", model, model_inner, (1, h, w), perm=None)
    # print("After export")
    # model.graph.print_jit_graph()
    b_in, b_out, acc = nemo.utils.get_intermediate_activations(model_inner, validate, val_loader)
    if acc != None:
        logging.info("After integerize: %.2f%%" % (100 * acc[0]))

    try:
        os.makedirs('HannaNet/golden')
    except Exception:
        pass

    from collections import OrderedDict
    dory_dict = OrderedDict([])
    # for key, value in b_in.items():
    #    print(key, value)
    # save super-node outputs as CSV files as golden reference
    bidx = 0
    # for x in model_inner.named_modules():
    #    print(x)
    for n, m in model_inner.named_modules():
        try:
            # print("n:%s" % n)
            # print("bidx:%d" % bidx)
            actbuf = b_in[n][0][bidx].permute((1, 2, 0))
        except RuntimeError:
            actbuf = b_in[n][0][bidx]
        np.savetxt("HannaNet/golden/golden_input_%s.txt" % n, actbuf.cpu().numpy().flatten(),
                   header="input (shape %s)" % (list(actbuf.shape)), fmt="%.3f", delimiter=',', newline=',\n')
    for n, m in model_inner.named_modules():
        try:
            actbuf = b_out[n][bidx].permute((1, 2, 0))
        except RuntimeError:
            actbuf = b_out[n][bidx]
        np.savetxt("HannaNet/golden/golden_%s.txt" % n, actbuf.cpu().numpy().flatten(),
                   header="%s (shape %s)" % (n, list(actbuf.shape)), fmt="%.3f", delimiter=',', newline=',\n')


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


    train_loader, validation_loader = LoadData(args)

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
        model = HannaNet(ConvBlock, [1, 1, 1], True)

    h = 96
    w = 160

    prec_dict = None
    # [NeMO] This used to preload the model with pretrained weights.
    if args.load_model is not None:

    # only use for running Quantize not TrainQuantized!!!!!
        model = nemo.transform.quantize_pact(model, dummy_input=torch.ones((1, 1, h, w)).to("cpu"))  # .cuda()
        logging.info("[ETHQ2] Model: %s", model)
        epochs, prec_dict = ModelManager.ReadQ(args.load_model, model)



    trainer = ModelTrainer(model, args, regime)
    if args.quantize:
        #trainer.TrainQuantized(train_loader, validation_loader, h, w, args.epochs)
        if prec_dict is not None:
            trainer.Quantize(validation_loader, h, w, prec_dict)
        else:
            trainer.Quantize(validation_loader, h, w)

        ExportONXX(model, model, validation_loader, trainer.ValidateSingleEpoch, h, w)


    if args.save_model is not None:
        # torch.save(trainer.model.state_dict(), args.save_model)
        ModelManager.Write(trainer.GetModel(), 100, args.save_model)

    print(model)


if __name__ == '__main__':
    main()