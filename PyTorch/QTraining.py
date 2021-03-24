
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

def LoadData(args):

    [x_train, x_validation, y_train, y_validation] = DataProcessor.ProcessTrainData(args.load_trainset)
    training_set = Dataset(x_train, y_train, True)
    validation_set = Dataset(x_validation, y_validation)

    # Parameters
    # num_workers - 0 for debug in Mac+PyCharm, 6 for everything else
    num_workers = 6
    params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': num_workers}
    train_loader = data.DataLoader(training_set, **params)
    validation_loader = data.DataLoader(validation_set, **params)

    return train_loader, validation_loader


# cmd arguments:
# --regime regime.json --epochs 10 --load-trainset "../Data/160x96OthersTrain.pickle" --load-model "../Models/PenguiNet160x96_32c.pt" --save-model "QModel.pt" --quantize


def main():

    Utils.Logger()

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch FrontNet')
    args = Utils.Parse(parser)

    torch.manual_seed(args.seed)

    # Load data
    train_loader, validation_loader = LoadData(args)

    model = ModelLib.PenguiNetModel(h=96, w=160, c=32, fc_nodes=1920)

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

    epoch = ModelManager.Read(args.load_model, model)
    trainer = ModelTrainer(model, args, regime)
    trainer.TrainQuantized(train_loader, validation_loader, h, w, args.epochs)

    if args.save_model is not None:
        ModelManager.Write(trainer.GetModel(), 100, args.save_model)

    print(model)



if __name__ == '__main__':
    main()