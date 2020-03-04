from __future__ import print_function
import logging
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from torch.utils import data
import sys
import matplotlib.patches as patches
import Utils as utils

sys.path.append("../PyTorch/")

from PreActBlock import PreActBlock
from Dronet import Dronet
from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from Dataset import Dataset
from ModelManager import ModelManager


def VizWorldTopView(frames, labels, camPoses, outputs, outputs2, isGray=False):

    fig = plt.figure(888, figsize=(15, 8))

    h = 9
    w = 16

    ax0 = plt.subplot2grid((h, w), (0, 0), colspan=7, rowspan=2)
    ax0.set_ylim([0, 9])
    annotation = ax0.annotate("poop", xy=(0, 5.5), size=10)
    annotation.set_animated(True)
    ax0.axis('off')

    ax1 = plt.subplot2grid((h, w), (2, 0), colspan=8, rowspan=7)
    ax1.set_title('World Frame Pose')
    plt.xlim(3.0, -3.0)
    plt.ylim(-3.0, 3.0)

    ax1.xaxis.tick_top()  # and move the X-Axis
    ax1.yaxis.tick_left()  # remove right y-Ticks
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_xlabel('Y')
    ax1.set_ylabel('X')

    plot1gt, = plt.plot([], [], color='green', label='GT', linestyle='None', marker='o', markersize=10)
    plot1pr, = plt.plot([], [], color='blue', label='Prediction', linestyle='None', marker='^', markersize=10)
    plot1pr2, = plt.plot([], [], color='red', label='Prediction Aug', linestyle='None', marker='D', markersize=10)
    plot1cam, = plt.plot([], [], color='k', label='Camera', linestyle='None', marker='s', markersize=10)
    arr1gt = ax1.arrow([], [], np.cos([]), np.sin([]), head_width=0.1, head_length=0.1, color='green', animated=True)
    arr1pr = ax1.arrow([], [], np.cos([]), np.sin([]), head_width=0.1, head_length=0.1, color='blue', animated=True)
    arr1pr2 = ax1.arrow([], [], np.cos([]), np.sin([]), head_width=0.1, head_length=0.1, color='red', animated=True)
    arr1cam = ax1.arrow([], [], np.cos([]), np.sin([]), head_width=0.1, head_length=0.1, color='k', animated=True)
    plt.plot([2.4, -2.4], [2.4, 2.4], color='gray', linestyle='solid')
    plt.plot([2.4, -2.4], [-2.4, -2.4], color='gray', linestyle='solid')

    plt.plot([2.4, 2.4], [2.4, -2.4], color='gray', linestyle='solid')
    plt.plot([-2.4, -2.4], [2.4, -2.4], color='gray', linestyle='solid')

    plt.plot([2.4, -2.4], [1.2, 1.2], color='gray', linestyle='solid')
    plt.plot([2.4, -2.4], [0, 0], color='gray', linestyle='solid')
    plt.plot([2.4, -2.4], [-1.2, -1.2], color='gray', linestyle='solid')

    plt.plot([1.2, 1.2], [2.4, -2.4], color='gray', linestyle='solid')
    plt.plot([0, 0], [2.4, -2.4], color='gray', linestyle='solid')
    plt.plot([-1.2, -1.2], [2.4, -2.4], color='gray', linestyle='solid')

    plt.legend(loc='lower right', bbox_to_anchor=(0.8, 0.2, 0.25, 0.25))

    ax3 = plt.subplot2grid((h, w), (2, 9), rowspan=7, colspan=7)
    ax3.axis('off')
    frame = frames[0].astype(np.uint8)
    if isGray == True:
        imgplot = plt.imshow(frame, cmap="gray", vmin=0, vmax=255)
    else:
        frame = frame.transpose(1, 2, 0)
        imgplot = plt.imshow(frame)

    ax4 = plt.subplot2grid((h, w), (0, 9), colspan=7)
    ax4.set_xlim([0, 8])
    annotation2 = ax4.annotate("poop", xy=(3, 0.1), size=14, weight='bold')
    annotation2.set_animated(True)
    ax4.axis('off')

    plt.subplots_adjust(wspace=1.5)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='FullMetalNicky'))

    def animate(id):

        label = labels[id]
        camPose = camPoses[id]
        pred = outputs[id]
        pred2 = outputs2[id]
        x_gt, y_gt, phi_gt = utils.MoveToWorldFrame(label, camPose)
        x_pred, y_pred, phi_pred = utils.MoveToWorldFrame(pred, camPose)
        x_pred2, y_pred2, phi_pred2 = utils.MoveToWorldFrame(pred2, camPose)

        x_cam, y_cam, z_cam, phi_cam = camPose[0], camPose[1], camPose[2], camPose[3]

        str1 = "x_cam={:05.3f}, y_cam={:05.3f}, phi_cam={:05.3f} {}".format(x_cam, y_cam, phi_cam, "\n")
        str1 = str1 + "x_gt={:05.3f}, y_gt={:05.3f}, phi_gt={:05.3f} {}".format(x_gt, y_gt, phi_gt, "\n")
        str1 = str1 + "x_pr={:05.3f}, y_pr={:05.3f},  phi_pr={:05.3f} {}".format(x_pred, y_pred, phi_pred, "\n")
        str1 = str1 + "Aug:x_pr={:05.3f}, y_pr={:05.3f},  phi_pr={:05.3f}".format(x_pred2, y_pred2, phi_pred2)

        annotation.set_text(str1)

        plot1gt.set_data(np.array([y_gt, x_gt]))
        plot1pr.set_data(np.array([y_pred, x_pred]))
        plot1pr2.set_data(np.array([y_pred2, x_pred2]))
        plot1cam.set_data(np.array([y_cam, x_cam]))

        if (len(ax1.patches) > 1):
            ax1.patches.pop()
            ax1.patches.pop()
            ax1.patches.pop()
            ax1.patches.pop()

        patch1 = patches.FancyArrow(y_gt, x_gt, 0.5 * np.sin(phi_gt), 0.5 * np.cos(phi_gt), head_width=0.05,
                                    head_length=0.05, color='green')
        patch2 = patches.FancyArrow(y_pred, x_pred, 0.5 * np.sin(phi_pred), 0.5 * np.cos(phi_pred), head_width=0.05,
                                    head_length=0.05, color='blue')
        patch3 = patches.FancyArrow(y_pred2, x_pred2, 0.5 * np.sin(phi_pred2), 0.5 * np.cos(phi_pred2), head_width=0.05,
                                    head_length=0.05, color='red')
        patch4 = patches.FancyArrow(y_cam, x_cam, 0.5 * np.sin(phi_cam), 0.5 * np.cos(phi_cam), head_width=0.05,
                                    head_length=0.05, color='k')

        ax1.add_patch(patch1)
        ax1.add_patch(patch2)
        ax1.add_patch(patch3)
        ax1.add_patch(patch4)

        frame = frames[id].astype(np.uint8)
        if isGray == False:
            frame = frame.transpose(1, 2, 0)
        imgplot.set_array(frame)

        annotation2.set_text('Frame {}'.format(id))

        # use the first one for viz on screen and second one for video recording
        # return plot1gt, plot1pr, patch1, patch2, scatter2gt, scatter2pr, imgplot, ax1, ax3, annotation, annotation2
        #return plot1gt, plot1pr, patch1, patch2, scatter2gt, scatter2pr, imgplot, annotation, annotation2

        return plot1gt, plot1pr, plot1pr2, plot1cam, patch1, patch2, patch3, patch4, imgplot, annotation, annotation2

    ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=1, blit=True)
    ani.save('balh.mp4', writer=writer)
    # ani.save('viz2.gif', dpi=80, writer='imagemagick')
    plt.show()



def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename="log.txt",
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    model = Dronet(PreActBlock, [1, 1, 1], True)
    ModelManager.Read('../PyTorch/Models/DronetGray.pt', model)

    DATA_PATH = "/Users/usi/PycharmProjects/data/"
    [x_test, y_test, z_test] = DataProcessor.ProcessTestData(DATA_PATH + "Nicewall2.pickle", 60, 108, True, True)
    t_test = DataProcessor.GetTimeStampsFromTestData(DATA_PATH + "Nicewall2.pickle")

    test_set = Dataset(x_test, y_test)
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 0}
    test_generator = data.DataLoader(test_set, **params)
    trainer = ModelTrainer(model)


    MSE, MAE, r2_score, outputs, gt_labels = trainer.Test(test_generator)

   # utils.SaveResultsToCSV(gt_labels, outputs, t_test, "wow.csv")
    utils.SaveResultsToCSVWithCamPoses(gt_labels, outputs, t_test, z_test, "wow.csv")

    model2 = Dronet(PreActBlock, [1, 1, 1], True)
    ModelManager.Read('../PyTorch/Models/DronetGrayAug120.pt', model2)
    trainer2 = ModelTrainer(model2)
    MSE, MAE, r2_score, outputs2, gt_labels2 = trainer2.Test(test_generator)
    outputs2 = np.reshape(outputs2, (-1, 4))

    x_test = np.reshape(x_test, (-1, 60, 108))
    VizWorldTopView(x_test, y_test, z_test, outputs, outputs2, True)

if __name__ == '__main__':
    main()