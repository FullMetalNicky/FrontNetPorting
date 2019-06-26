import torch.nn as nn
import torch
import math
import numpy as np
from ValidationUtils import RunningAverage
from ValidationUtils import MovingAverage
from ModelManager import ModelManager
from DataVisualization import DataVisualization


class ModelTrainer:
    def __init__(self, model, num_epochs=80):
        self.num_epochs = num_epochs
        self.learning_rate = 0.001
        self.batch_size = 64
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model.to(self.device)
        self.model_manager = ModelManager()
        self.visualizer = DataVisualization()

        # Loss and optimizer
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=np.sqrt(0.1), patience=5, verbose=False,
                                                   threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.1e-6,
                                                   eps=1e-08)

    def update_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def augmentor(self, sample, target):
        frame = sample
        if np.random.choice([True, False]):
            frame = torch.flip(frame, [1])
            target[1] = -target[1]  # Y
            target[3] = -target[3]  # Relative YAW
        return frame, target


    def Train(self, training_generator, validation_generator):
        train_losses = []
        valid_losses = []
        MSEs = []
        y_pred_viz = []
        gt_labels_viz = []

        for epoch in range(self.num_epochs):
            print("Starting Epoch {}".format(epoch + 1))

            self.model.train()
            train_loss = MovingAverage()
            i = 0

            for batch_samples, batch_targets in training_generator:

                batch_targets = batch_targets.to(self.device)
                batch_samples = batch_samples.to(self.device)
                outputs = self.model(batch_samples)
                loss = self.criterion(outputs, batch_targets)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss.update(loss)

                if (i + 1) % 100 == 0:
                    print("Epoch [{}/{}], Step [{}] Loss: {:.4f}"
                          .format(epoch + 1, self.num_epochs, i + 1, loss.item()))

                i += 1
            train_losses.append(train_loss.value)

            self.model.eval()
            valid_loss = RunningAverage()
            y_pred = []
            gt_labels = []
            with torch.no_grad():
                for batch_samples, batch_targets in validation_generator:
                    gt_labels.extend(batch_targets.cpu().numpy())
                    batch_targets = batch_targets.to(self.device)
                    batch_samples = batch_samples.to(self.device)
                    outputs = self.model(batch_samples)
                    loss = self.criterion(outputs, batch_targets)
                    valid_loss.update(loss)
                    y_pred.extend(outputs.cpu().numpy())

                print("Average loss {}".format(valid_loss))

            self.scheduler.step(valid_loss)

            gt_labels = torch.tensor(gt_labels, dtype=torch.float32)
            y_pred = torch.tensor(y_pred, dtype=torch.float32)
            MSE = torch.sqrt(torch.mean((y_pred - gt_labels).pow(2), 0))
            MSEs.append(MSE)
            y_pred_viz.append(y_pred)
            gt_labels_viz.append(gt_labels)
            print('Validation MSE: {}'.format(MSE))
            valid_losses.append(valid_loss.value)
            checkpoint_filename = 'FrontNet-{:03d}.pkl'.format(epoch)
            self.model_manager.Write(self.optimizer, self.model, epoch, checkpoint_filename)

        self.visualizer.PlotLoss(train_losses, valid_losses)
        self.visualizer.PlotMSE(MSEs)
        self.visualizer.PlotGTandEstimationVsTime(gt_labels_viz, y_pred_viz)
        self.visualizer.PlotGTVsEstimation(gt_labels_viz, y_pred_viz)
        self.visualizer.DisplayPlots()
        return train_losses, valid_losses

    def PerdictSingleSample(self, test_generator):

        iterator = iter(test_generator)
        batch_samples, batch_targets = iterator.next()
        index = np.random.choice(np.arange(0, batch_samples.shape[0]), 1)
        x_test = batch_samples[index]
        y_test = batch_targets[index]
        self.model.eval()

        print('GT Values: {}'.format(y_test.cpu().numpy()))
        with torch.no_grad():
            x_test = x_test.to(self.device)
            outputs = self.model(x_test)
        print('Prediction Values: {}'.format(outputs.cpu().numpy()))


    def Predict(self, test_generator):

        valid_loss = RunningAverage()
        y_pred = []
        gt_labels = []
        self.model.eval()

        with torch.no_grad():
            for batch_samples, batch_targets in test_generator:

                print('GT Values: {}'.format(batch_targets.cpu().numpy()))
                gt_labels.extend(batch_targets.cpu().numpy())
                batch_targets = batch_targets.to(self.device)
                batch_samples = batch_samples.to(self.device)
                outputs = self.model(batch_samples)
                loss = self.criterion(outputs, batch_targets)
                valid_loss.update(loss)
                y_pred.extend(outputs.cpu().numpy())
                print('Prediction Values: {}'.format(outputs.cpu().numpy()))

            print("Average loss {}".format(valid_loss))

        gt_labels = torch.tensor(gt_labels, dtype=torch.float32)
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
        MSE = torch.sqrt(torch.mean((y_pred - gt_labels).pow(2), 0))
        print('Validation MSE: {}'.format(MSE))