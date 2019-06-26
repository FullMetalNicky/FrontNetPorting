
import matplotlib.pyplot as plt
import torch
import matplotlib.gridspec as gridspec
import numpy as np

class DataVisualization:
    def __init__(self):
        self.figure_counter = 0

    def PlotLoss(self, train_losses, valid_losses):
        self.figure_counter += 1
        plt.figure(self.figure_counter, figsize=(10, 6))

        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, color='green',  label='Training loss')
        plt.plot(epochs, valid_losses, color='blue', label='Validation loss')
        plt.legend()
        plt.title('Learning curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(epochs)


    def PlotMSE(self, MSE):
        self.figure_counter += 1
        plt.figure(self.figure_counter, figsize=(10, 6))

        epochs = range(1, len(MSE) + 1)
        MSE = torch.stack(MSE, 0)
        x = MSE[:, 0]
        x = x.cpu().numpy()
        plt.plot(epochs, x, color='green', label='x')
        y = MSE[:, 1]
        y = y.cpu().numpy()
        plt.plot(epochs, y, color='blue', label='y')
        z = MSE[:, 2]
        z = z.cpu().numpy()
        plt.plot(epochs, z, color='r', marker='o', label='z')
        phi = MSE[:, 3]
        phi = phi.cpu().numpy()
        plt.plot(epochs, phi, color='m', marker='o', label='phi')
        plt.legend()
        plt.title('Pose Variables MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.xticks(epochs)


    def PlotGTandEstimationVsTime(self, gt_labels, predictions):
        self.figure_counter += 1
        plt.figure(self.figure_counter, figsize=(10, 6))

        gt_labels = torch.stack(gt_labels, 0)
        predictions = torch.stack(predictions, 0)
        gt_labels = gt_labels.cpu().numpy()
        gt_labels = np.reshape(gt_labels, (-1, 4))
        predictions = predictions.cpu().numpy()
        predictions = np.reshape(predictions, (-1, 4))
        samples = len(gt_labels[:, 0])
        samples = range(1, samples+1)

        gs = gridspec.GridSpec(2, 2)
        ax = plt.subplot(gs[0, 0])
        ax.set_title('x')
        x_gt = gt_labels[:, 0]
        x_pred = predictions[:, 0]
        plt.plot(samples, x_gt, color='green', label='GT')
        plt.plot(samples, x_pred, color='black', label='Prediction')
        plt.legend()

        ax = plt.subplot(gs[0, 1])
        ax.set_title('y')
        y_gt = gt_labels[:, 1]
        y_pred = predictions[:, 1]
        plt.plot(samples, y_gt, color='blue', label='GT')
        plt.plot(samples, y_pred, color='black', label='Prediction')
        plt.legend()

        ax = plt.subplot(gs[1, 0])
        ax.set_title('z')
        z_gt = gt_labels[:, 2]
        z_pred = predictions[:, 2]
        plt.plot(samples, z_gt, color='r', label='GT')
        plt.plot(samples, z_pred, color='black', label='Prediction')
        plt.legend()

        ax = plt.subplot(gs[1, 1])
        ax.set_title('phi')
        phi_gt = gt_labels[:, 3]
        phi_pred = predictions[:, 3]
        plt.plot(samples, phi_gt, color='m', label='GT')
        plt.plot(samples, phi_pred, color='black', label='Prediction')
        plt.legend()

        plt.subplots_adjust(hspace=0.3)
        plt.suptitle('Ground Truth and Predictions vs time')

    def DisplayPlots(self):
        plt.show()

    def PlotGTVsEstimation(self, gt_labels, predictions):
        self.figure_counter += 1
        plt.figure(self.figure_counter, figsize=(10, 6))

        gt_labels = torch.stack(gt_labels, 0)
        predictions = torch.stack(predictions, 0)
        gt_labels = gt_labels.cpu().numpy()
        gt_labels = np.reshape(gt_labels, (-1, 4))
        predictions = predictions.cpu().numpy()
        predictions = np.reshape(predictions, (-1, 4))

        gs = gridspec.GridSpec(2, 2)
        ax = plt.subplot(gs[0, 0])
        ax.set_title('x')
        x_gt = gt_labels[:, 0]
        x_pred = predictions[:, 0]
        plt.scatter(x_gt, x_pred, color='green', marker='o')
        plt.legend()

        ax = plt.subplot(gs[0, 1])
        ax.set_title('y')
        y_gt = gt_labels[:, 1]
        y_pred = predictions[:, 1]
        plt.scatter(y_gt, y_pred, color='blue', marker='o')
        plt.legend()

        ax = plt.subplot(gs[1, 0])
        ax.set_title('z')
        z_gt = gt_labels[:, 2]
        z_pred = predictions[:, 2]
        plt.scatter(z_gt, z_pred, color='r', marker='o')
        plt.legend()

        ax = plt.subplot(gs[1, 1])
        ax.set_title('phi')
        phi_gt = gt_labels[:, 3]
        phi_pred = predictions[:, 3]
        plt.scatter(phi_gt, phi_pred, color='m', marker='o')
        plt.legend()

        plt.subplots_adjust(hspace=0.3)
        plt.suptitle('Ground Truth vs Predictions')
