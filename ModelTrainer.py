import torch.nn as nn
import torch
import numpy as np
from ValidationUtils import RunningAverage
from ValidationUtils import MovingAverage
from DataVisualization import DataVisualization
from EarlyStopping import EarlyStopping
from ValidationUtils import Metrics


class ModelTrainer:
    def __init__(self, model, num_epochs=80):
        self.num_epochs = num_epochs
        self.learning_rate = 0.001
        self.batch_size = 64
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model.to(self.device)
        self.visualizer = DataVisualization()

        # Loss and optimizer
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=np.sqrt(0.1), patience=5, verbose=False,
                                                   threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0.1e-6,
                                                   eps=1e-08)
        self.early_stopping = EarlyStopping(patience=10, verbose=True)
        self.metrics = Metrics()


    def Train(self, training_generator, validation_generator):
        train_losses = []
        valid_losses = []
        y_pred_viz = []
        gt_labels_viz = []
        self.metrics.Reset()

        for epoch in range(self.num_epochs):
            print("Starting Epoch {}".format(epoch + 1))

            self.model.train()
            train_loss = MovingAverage()
            i = 0

            for batch_samples, batch_targets in training_generator:

                batch_targets = batch_targets.to(self.device)
                batch_samples = batch_samples.to(self.device)
                outputs = self.model(batch_samples)

                loss_x = self.criterion(outputs[0], (batch_targets[:, 0]).view(-1, 1))
                loss_y = self.criterion(outputs[1], (batch_targets[:, 1]).view(-1, 1))
                loss_z = self.criterion(outputs[2], (batch_targets[:, 2]).view(-1, 1))
                loss_phi = self.criterion(outputs[3], (batch_targets[:, 3]).view(-1, 1))
                loss = loss_x + loss_y + loss_z + loss_phi

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
                    #gt_labels.extend(batch_targets)
                    batch_targets = batch_targets.to(self.device)
                    batch_samples = batch_samples.to(self.device)
                    outputs = self.model(batch_samples)

                    loss_x = self.criterion(outputs[0], (batch_targets[:, 0]).view(-1, 1))
                    loss_y = self.criterion(outputs[1], (batch_targets[:, 1]).view(-1, 1))
                    loss_z = self.criterion(outputs[2], (batch_targets[:, 2]).view(-1, 1))
                    loss_phi = self.criterion(outputs[3], (batch_targets[:, 3]).view(-1, 1))
                    loss = loss_x + loss_y + loss_z + loss_phi

                    valid_loss.update(loss)
                    outputs = torch.stack(outputs, 0)
                    outputs = torch.squeeze(outputs)
                    outputs = torch.t(outputs)
                    y_pred.extend(outputs.cpu().numpy())
                    #y_pred.extend(outputs.cpu())

                print("Average loss {}".format(valid_loss))

            self.scheduler.step(valid_loss)

            gt_labels = torch.tensor(gt_labels, dtype=torch.float32)
            y_pred = torch.tensor(y_pred, dtype=torch.float32)
            MSE, MAE, r_score = self.metrics.Update(y_pred, gt_labels)

            y_pred_viz.append(y_pred)
            gt_labels_viz.append(gt_labels)
            print('Validation MSE: {}'.format(MSE))
            print('Validation MAE: {}'.format(MAE))
            print('Test r_score: {}'.format(r_score))


            valid_losses.append(valid_loss.value)
            checkpoint_filename = 'FrontNet-{:03d}.pkl'.format(epoch)
            self.early_stopping(valid_loss.value, self.model, epoch, checkpoint_filename, self.optimizer)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        MSEs = self.metrics.GetMSE()
        MAEs = self.metrics.GetMAE()
        r_score = self.metrics.Getr2_score()
        #score = self.metrics.Getr2_score(gt_labels_viz, y_pred_viz)
        #print('Validation r^2 score: {}'.format(score))

        self.visualizer.PlotLoss(train_losses, valid_losses)
        self.visualizer.PlotMSE(MSEs)
        self.visualizer.PlotMAE(MAEs)
        self.visualizer.PlotR2Score(r_score)

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

        #self.visualizer.DisplayVideoFrame(x_test[0].cpu().numpy())

        print('GT Values: {}'.format(y_test.cpu().numpy()))
        with torch.no_grad():
            x_test = x_test.to(self.device)
            outputs = self.model(x_test)

        outputs = torch.stack(outputs, 0)
        outputs = torch.squeeze(outputs)
        outputs = torch.t(outputs)
        print('Prediction Values: {}'.format(outputs.cpu().numpy()))


    def Predict(self, test_generator):

        valid_loss = RunningAverage()
        y_pred = []
        gt_labels = []
        y_pred_viz = []
        gt_labels_viz = []
        self.model.eval()
        self.metrics.Reset()

        with torch.no_grad():
            for batch_samples, batch_targets in test_generator:

                gt_labels.extend(batch_targets.cpu().numpy())
                batch_targets = batch_targets.to(self.device)
                batch_samples = batch_samples.to(self.device)
                outputs = self.model(batch_samples)

                loss_x = self.criterion(outputs[0], (batch_targets[:, 0]).view(-1, 1))
                loss_y = self.criterion(outputs[1], (batch_targets[:, 1]).view(-1, 1))
                loss_z = self.criterion(outputs[2], (batch_targets[:, 2]).view(-1, 1))
                loss_phi = self.criterion(outputs[3], (batch_targets[:, 3]).view(-1, 1))
                loss = loss_x + loss_y + loss_z + loss_phi

                valid_loss.update(loss)
                outputs = torch.stack(outputs, 0)
                outputs = torch.squeeze(outputs)
                outputs = torch.t(outputs)
                y_pred.extend(outputs.cpu().numpy())

            print("Average loss {}".format(valid_loss))

        gt_labels = torch.tensor(gt_labels, dtype=torch.float32)
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
        MSE, MAE, r_score = self.metrics.Update(y_pred, gt_labels)

        y_pred_viz.append(y_pred)
        gt_labels_viz.append(gt_labels)
        self.visualizer.PlotGTandEstimationVsTime(gt_labels_viz, y_pred_viz)
        self.visualizer.PlotGTVsEstimation(gt_labels_viz, y_pred_viz)
        self.visualizer.DisplayPlots()
        print('Test MSE: {}'.format(MSE))
        print('Test MAE: {}'.format(MAE))
        print('Test r_score: {}'.format(r_score))

