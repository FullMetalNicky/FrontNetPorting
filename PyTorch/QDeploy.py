
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
# --regime regime.json --epochs 10 --load-trainset "../Data/160x96OthersTrain.pickle" --load-model "../Models/QModel.pth" --quantize


def main():

    Utils.Logger()

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch FrontNet')
    args = Utils.Parse(parser)

    torch.manual_seed(args.seed)

    # Load data
    [x_train, x_validation, y_train, y_validation] = DataProcessor.ProcessTrainData(args.load_trainset)
    validation_set = Dataset(x_validation, y_validation)
    num_workers = 6
    params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': num_workers}
    validation_loader = data.DataLoader(validation_set, **params)


    model = ModelLib.PenguiNetModel()

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
    trainer.Deploy(validation_loader, h, w, prec_dict)

    if args.save_model is not None:
        ModelManager.Write(trainer.GetModel(), 100, args.save_model)

    print(model)



if __name__ == '__main__':
    main()