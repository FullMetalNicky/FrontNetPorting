
import matplotlib.pyplot as plt
import torch
import matplotlib.gridspec as gridspec
import numpy as np
import cv2

class DataVisualization:

    figure_counter = 0
    folderPath = "Results/"
    desc = ""

    @staticmethod
    def PlotLoss(train_losses_x, train_losses_y, train_losses_z, train_losses_phi , valid_losses_x, valid_losses_y, valid_losses_z, valid_losses_phi):

        epochs = range(1, len(train_losses_x) + 1)

        DataVisualization.figure_counter += 1
        plt.figure(DataVisualization.figure_counter, figsize=(20, 12))
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

        plt.savefig(DataVisualization.folderPath + DataVisualization.desc +'LearningCurves.png')


    @staticmethod
    def PlotMSE(MSE):
        DataVisualization.figure_counter += 1
        plt.figure(DataVisualization.figure_counter, figsize=(10, 6))

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
        plt.savefig(DataVisualization.folderPath + DataVisualization.desc + 'MSE.png')



    @staticmethod
    def PlotMAE(MAE):
        DataVisualization.figure_counter += 1
        plt.figure(DataVisualization.figure_counter, figsize=(10, 6))

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
        plt.savefig(DataVisualization.folderPath + DataVisualization.desc + 'MAE.png')


    @staticmethod
    def PlotR2Score(r2_score):
        DataVisualization.figure_counter += 1
        plt.figure(DataVisualization.figure_counter, figsize=(10, 6))

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
        plt.savefig(DataVisualization.folderPath + DataVisualization.desc + 'Rsq.png')


    @staticmethod
    def PlotGTandEstimationVsTime(gt_labels, predictions):
        DataVisualization.figure_counter += 1
        plt.figure(DataVisualization.figure_counter, figsize=(20, 12))
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
        plt.savefig(DataVisualization.folderPath + DataVisualization.desc + 'GTandPredVsTime.png')


    @staticmethod
    def DisplayPlots():
        plt.show()


    @staticmethod
    def PlotGTVsEstimation(gt_labels, predictions):
        DataVisualization.figure_counter += 1
        plt.figure(DataVisualization.figure_counter, figsize=(20, 12))

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
        plt.savefig(DataVisualization.folderPath + DataVisualization.desc + 'GTvsPred.png')



    @staticmethod
    def DisplayFrameAndPose(frame, gt_labels, predictions):
        DataVisualization.figure_counter += 1
        plt.figure(DataVisualization.figure_counter, figsize=(10, 6))

        w = 20
        h = 12
        bar_length = h - 2
        offset_x = int((w-bar_length)/2)
        ax1 = plt.subplot2grid((h, w), (0, offset_x), colspan=bar_length)
        ax1.set_title('x')
        ax1.xaxis.tick_top()
        x_gt = gt_labels[0]
        x_pred = predictions[0]
        ax1.set_xlim([0, 4])
        ax1.set_ylim([-0.5, 0.5])
        ax1.set_yticklabels([])
        plt.scatter(x_gt, 0,  color='green', label='GT', s=100)
        plt.scatter(x_pred, 0, color='blue', label='Prediction', s=100)

        ax2 = plt.subplot2grid((h, w), (1, 0), rowspan=bar_length)
        ax2.set_title('y')
        y_gt = gt_labels[1]
        y_pred = predictions[1]
        ax2.set_ylim([-1, 1])
        ax2.set_xlim([-0.5, 0.5])
        ax2.set_xticklabels([])
        plt.scatter(0, y_gt, color='green', label='GT', s=100)
        plt.scatter(0, y_pred, color='blue', label='Prediction', s=100)

        ax3 = plt.subplot2grid((h, w), (1, 1), rowspan=bar_length, colspan=(w-2))
        ax3.set_yticklabels([])
        ax3.set_xticklabels([])
        frame = frame.transpose(1, 2, 0)
        frame = frame.astype(np.uint8)
        plt.imshow(frame)


        ax4 = plt.subplot2grid((h, w), (1, w-1), rowspan=bar_length)
        ax4.set_title('z')
        z_gt = gt_labels[2]
        z_pred = predictions[2]
        ax4.yaxis.tick_right()
        ax4.set_ylim([-1, 1])
        ax4.set_xlim([-0.5, 0.5])
        ax4.set_xticklabels([])
        plt.scatter(0, z_gt, color='green', label='GT', s=100)
        plt.scatter(0, z_pred, color='blue', label='Prediction', s=100)

        ax5 = plt.subplot2grid((h, w), (h-1, offset_x), colspan=bar_length)
        ax5.set_title('phi')
        phi_gt = gt_labels[3]
        phi_pred = predictions[3]
        ax5.set_xlim([-2, 2])
        ax5.set_ylim([-0.5, 0.5])
        ax5.set_yticklabels([])
        plt.scatter(phi_gt, 0, color='green', label='GT', s=100)
        plt.scatter(phi_pred, 0,  color='blue', label='Prediction', s=100)

        plt.subplots_adjust(hspace=0.3, wspace = 0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(DataVisualization.folderPath + DataVisualization.desc + 'GTandPredandPose.png')



    @staticmethod
    def DisplayVideoFrame(frame):

        frame = frame.transpose(1, 2, 0)
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('frame', frame)
        cv2.waitKey(10)


    @staticmethod
    def DisplayDatasetVideo(data):

        length = len(data)
        for i in range(0, length):
            DataVisualization.DisplayVideoFrame(data[i])

