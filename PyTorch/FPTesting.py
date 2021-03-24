
from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from Dataset import Dataset
from torch.utils import data
from ModelManager import ModelManager
import ModelLib
import Utils


def main():
    Utils.Logger()

    # Load the training data (which will be split to validation and train)
    testDataPath = "../Data/160x96OthersTest.pickle"
    [x_test, y_test] = DataProcessor.ProcessTestData(testDataPath)

    # Create the PyTorch data loaders
    test_set = Dataset(x_test, y_test)
    params = {'batch_size': 64, 'shuffle': False, 'num_workers': 1}
    test_loader = data.DataLoader(test_set, **params)

    # Choose your model
    model = ModelLib.PenguiNetModel(h=96, w=160, c=32, fc_nodes=1920)
    ModelManager.Read("../Models/PenguiNet160x96_32c.pt", model)

    # This class handles the training loop.
    # It reads the training params from Francesco's .json or uses default values
    # So you need to run the script with an argument in the cmd
    trainer = ModelTrainer(model)
    MSE, MAE, r2_score, outputs, labels = trainer.Test(test_loader)


if __name__ == '__main__':
    main()