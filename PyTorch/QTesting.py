
from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from Dataset import Dataset
from ModelManager import ModelManager
import ModelLib
from torch.utils import data
import Utils
import argparse
import json
import torch
import logging
import nemo



# cmd arguments:
# --regime regime.json --epochs 10 --load-testset "../Data/160x96OthersTest.pickle" --load-model "../Models/QModel.pth" --quantize


def main():

    Utils.Logger()

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch FrontNet')
    args = Utils.Parse(parser)

    torch.manual_seed(args.seed)

    # Load data
    [x_test, y_test] = DataProcessor.ProcessTestData(args.load_testset)

    # Create the PyTorch data loaders
    test_set = Dataset(x_test, y_test)
    params = {'batch_size': 64, 'shuffle': False, 'num_workers': 1}
    test_loader = data.DataLoader(test_set, **params)

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

    h = 96
    w = 160

    model = ModelLib.PenguiNetModel(h=96, w=160, c=32, fc_nodes=1920)

    model = nemo.transform.quantize_pact(model, dummy_input=torch.ones((1, 1, h, w)).to("cpu"))
    logging.info("[ETHQ] Model: %s", model)
    epoch, prec_dict = ModelManager.ReadQ(args.load_model, model)
    trainer = ModelTrainer(model, args, regime)
    trainer.Test(test_loader)




if __name__ == '__main__':
    main()