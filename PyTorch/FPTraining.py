
from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from Dataset import Dataset
import ModelLib
from torch.utils import data
import Utils



def main():

    Utils.Logger()

    # Load the training data (which will be split to validation and train)
    trainDataPath = "../Data/160x96OthersTrain.pickle"
    [x_train, x_validation, y_train, y_validation] = DataProcessor.ProcessTrainData(trainDataPath)

    # Create the PyTorch data loaders
    params = {'batch_size': 64, 'shuffle': True, 'num_workers': 6}
    training_set = Dataset(x_train, y_train, train=True)
    training_generator = data.DataLoader(training_set, **params)
    validation_set = Dataset(x_validation, y_validation, train=False)
    validation_generator = data.DataLoader(validation_set, **params)

    # Choose your model
    model = ModelLib.PenguiNetModel(h=96, w=160, c=32, fc_nodes=1920)

    # This class handles the training loop
    trainer = ModelTrainer(model)
    trainer.Train(training_generator, validation_generator)



if __name__ == '__main__':
    main()