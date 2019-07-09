
import matplotlib.pyplot as plt
import torch
import matplotlib.gridspec as gridspec
import numpy as np
import cv2

class DataVisualization:
    def __init__(self):
        self.figure_counter = 0

    def PlotLoss(self, train_losses_x, train_losses_y, train_losses_z, train_losses_phi , valid_losses_x, valid_losses_y, valid_losses_z, valid_losses_phi):
        #self.figure_counter += 1
        #plt.figure(self.figure_counter, figsize=(10, 6))

        epochs = range(1, len(train_losses_x) + 1)
       # plt.plot(epochs, train_losses, color='green',  label='Training loss')
        #plt.plot(epochs, valid_losses, color='blue', label='Validation loss')

        self.figure_counter += 1
        plt.figure(self.figure_counter, figsize=(20, 12))
        plt.margins(0.1)

        gs = gridspec.GridSpec(2, 2)
        ax = plt.subplot(gs[0, 0])
        ax.set_title('x')


        plt.plot(epochs, train_losses_x, color='green', label='Training loss')
        plt.plot(epochs, valid_losses_x, color='black', label='Validation loss')
        plt.legend()

        ax = plt.subplot(gs[0, 1])
        ax.set_title('y')

        plt.plot(epochs, train_losses_y, color='blue', label='Training loss')
        plt.plot(epochs, valid_losses_y, color='black', label='Validation loss')
        plt.legend()

        ax = plt.subplot(gs[1, 0])
        ax.set_title('z')

        plt.plot(epochs, train_losses_z, color='r', label='Training loss')
        plt.plot(epochs, valid_losses_z, color='black', label='Validation loss')
        plt.legend()

        ax = plt.subplot(gs[1, 1])
        ax.set_title('phi')

        plt.plot(epochs, train_losses_phi, color='m', label='Training loss')
        plt.plot(epochs, valid_losses_phi, color='black', label='Validation loss')
        plt.legend()

        plt.subplots_adjust(hspace=0.3)
        plt.suptitle('Learning curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')



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


    def PlotMAE(self, MAE):
        self.figure_counter += 1
        plt.figure(self.figure_counter, figsize=(10, 6))

        epochs = range(1, len(MAE) + 1)
        MAE = torch.stack(MAE, 0)
        x = MAE[:, 0]
        x = x.cpu().numpy()
        plt.plot(epochs, x, color='green', label='x')
        y = MAE[:, 1]
        y = y.cpu().numpy()
        plt.plot(epochs, y, color='blue', label='y')
        z = MAE[:, 2]
        z = z.cpu().numpy()
        plt.plot(epochs, z, color='r', marker='o', label='z')
        phi = MAE[:, 3]
        phi = phi.cpu().numpy()
        plt.plot(epochs, phi, color='m', marker='o', label='phi')
        plt.legend()
        plt.title('Pose Variables MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.xticks(epochs)

    def PlotR2Score(self, r2_score):
        self.figure_counter += 1
        plt.figure(self.figure_counter, figsize=(10, 6))

        epochs = range(1, len(r2_score) + 1)
        r2_score = torch.stack(r2_score, 0)
        x = r2_score[:, 0]
        x = x.cpu().numpy()
        plt.plot(epochs, x, color='green', label='x')
        y = r2_score[:, 1]
        y = y.cpu().numpy()
        plt.plot(epochs, y, color='blue', label='y')
        z = r2_score[:, 2]
        z = z.cpu().numpy()
        plt.plot(epochs, z, color='r', marker='o', label='z')
        phi = r2_score[:, 3]
        phi = phi.cpu().numpy()
        plt.plot(epochs, phi, color='m', marker='o', label='phi')
        plt.legend()
        plt.title('Pose Variables r2_score')
        plt.xlabel('Epoch')
        plt.ylabel('r2_score')
        plt.xticks(epochs)


    def PlotGTandEstimationVsTime(self, gt_labels, predictions):
        self.figure_counter += 1
        plt.figure(self.figure_counter, figsize=(20, 12))
        plt.margins(0.1)

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
        plt.figure(self.figure_counter, figsize=(20, 12))

        gt_labels = torch.stack(gt_labels, 0)
        predictions = torch.stack(predictions, 0)
        gt_labels = gt_labels.cpu().numpy()
        gt_labels = np.reshape(gt_labels, (-1, 4))
        predictions = predictions.cpu().numpy()
        predictions = np.reshape(predictions, (-1, 4))

        gs = gridspec.GridSpec(2, 2)
        ax = plt.subplot(gs[0, 0])
        ax.set_title('x')
        ax.set_xmargin(0.2)
        x_gt = gt_labels[:, 0]
        x_pred = predictions[:, 0]
        plt.scatter(x_gt, x_pred, color='green', marker='o')
        plt.plot(x_gt, x_gt, color='black', linestyle='--')

        plt.legend()

        ax = plt.subplot(gs[0, 1])
        ax.set_title('y')
        ax.set_xmargin(0.2)
        y_gt = gt_labels[:, 1]
        y_pred = predictions[:, 1]
        plt.scatter(y_gt, y_pred, color='blue', marker='o')
        plt.plot(y_gt, y_gt, color='black', linestyle='--')
        plt.legend()

        ax = plt.subplot(gs[1, 0])
        ax.set_title('z')
        ax.set_xmargin(0.2)
        z_gt = gt_labels[:, 2]
        z_pred = predictions[:, 2]
        plt.scatter(z_gt, z_pred, color='r', marker='o')
        plt.plot(z_gt, z_gt, color='black', linestyle='--')

        plt.legend()

        ax = plt.subplot(gs[1, 1])
        ax.set_title('phi')
        ax.set_xmargin(0.2)
        phi_gt = gt_labels[:, 3]
        phi_pred = predictions[:, 3]
        plt.scatter(phi_gt, phi_pred, color='m', marker='o')
        plt.plot(phi_gt, phi_gt, color='black', linestyle='--')
        plt.legend()

        plt.subplots_adjust(hspace=0.3)
        plt.suptitle('Ground Truth vs Predictions')


    def DisplayVideoFrame(self, frame):

        frame = frame.transpose(1, 2, 0)
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('frame', frame)
        cv2.waitKey(10)


    def DisplayDatasetVideo(self, data):

        length = len(data)
        for i in range(0, length):
            self.DisplayVideoFrame(data[i])