from __future__ import print_function
from FrontNet import PreActBlock
from FrontNet import FrontNet
from DataProcessor import DataProcessor
from ModelTrainer import ModelTrainer
from Dataset import Dataset
from torch.utils import data
from ModelManager import ModelManager
from DataVisualization import DataVisualization
import logging
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.append("../pulp/")
from ImageIO import ImageIO


import matplotlib.patches as patches


def VizDroneBEV(frames, labels, outputs):

    fig = plt.figure(888, figsize=(15, 5))

    h = 9
    w = 16

    ax0 = plt.subplot2grid((h, w), (0, 0), colspan=7, rowspan=2)
    ax0.set_ylim([0, 9])
    annotation = ax0.annotate("poop", xy=(0, 5.5), size=10)
    annotation.set_animated(True)
    ax0.axis('off')

    ax1 = plt.subplot2grid((h, w), (2, 0), colspan=8, rowspan=7)
    ax1.set_title('Relative Pose (x,y)')
    ax1.yaxis.set_ticks([0, 1.5, 3])  # set y-ticks
    ax1.xaxis.set_ticks([-3.0, -1.5, 0, 1.5, 3.0])  # set y-ticks
    ax1.xaxis.tick_top()  # and move the X-Axis
    ax1.yaxis.tick_left()  # remove right y-Ticks
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    trianglex = [3, 0, -3, 3]
    triangley = [3, 0, 3, 3]
    collection = plt.fill(trianglex, triangley, facecolor='lightskyblue')

    plot1gt, = plt.plot([], [], color='green', label='Bebop', linestyle='None', marker='o', markersize=10)
    plot1pr, = plt.plot([], [], color='blue', label='Himax', linestyle='None', marker='o', markersize=10)
    arr1gt = ax1.arrow([], [], np.cos([]), np.sin([]), head_width=0.1, head_length=0.1, color='green', animated=True)
    arr1pr = ax1.arrow([], [], np.cos([]), np.sin([]), head_width=0.1, head_length=0.1, color='blue', animated=True)
    plt.legend(loc='lower right', bbox_to_anchor=(0.8, 0.2, 0.25, 0.25))

    ax2 = plt.subplot2grid((h, w), (2, 8), rowspan=7)
    ax2.set_title('Relative z', pad=20)
    ax2.yaxis.tick_right()
    ax2.set_xlim([-0.5, 0.5])
    ax2.set_xticklabels([])
    ax2.yaxis.set_ticks([-1, 0, 1])  # set y-ticks
    ax2.xaxis.set_ticks_position('none')
    scatter2gt = plt.scatter([], [], color='green', label='Bebop', s=100)
    scatter2pr = plt.scatter([], [], color='blue', label='Himax', s=100)

    ax3 = plt.subplot2grid((h, w), (2, 9), rowspan=7, colspan=7)
    ax3.axis('off')
    #frame = frames[0].transpose(1, 2, 0)
    #frame = frame[:, :, 0]
    frame = frames[0].astype(np.uint8)
    imgplot = plt.imshow(frame,cmap="gray", vmin=0, vmax=255)

    ax4 = plt.subplot2grid((h, w), (0, 9), colspan=7)
    ax4.set_xlim([0, 8])
    annotation2 = ax4.annotate("poop", xy=(3, 0.1), size=14, weight='bold')
    annotation2.set_animated(True)
    ax4.axis('off')

    plt.subplots_adjust(wspace=1.5)

    img = mpimg.imread('minidrone.jpg')
    newax = fig.add_axes([0.26, 0.0, 0.1, 0.1], anchor='S')
    newax.imshow(img)
    newax.axis('off')

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=2, metadata=dict(artist='FullMetalNicky'))


    def animate(id):
        label = labels[id]
        output = outputs[id]
        x_gt, y_gt, z_gt, phi_gt = label[0], label[1], label[2], label[3]
        x_pred, y_pred, z_pred, phi_pred = output[0], output[1], output[2], output[3]

        str1 = "x_b={:05.3f}, y_b={:05.3f}, z_b={:05.3f}, phi_b={:05.3f} {}".format(x_gt, y_gt, z_gt, phi_gt, "\n")
        str1 = str1 + "x_h={:05.3f}, y_h={:05.3f}, z_h={:05.3f}, phi_h={:05.3f}".format(x_pred, y_pred, z_pred, phi_pred)

        phi_gt = phi_gt - np.pi / 2
        phi_pred = phi_pred - np.pi / 2

        annotation.set_text(str1)

        plot1gt.set_data(np.array([y_gt, x_gt]))
        plot1pr.set_data(np.array([y_pred, x_pred]))

        if(len(ax1.patches) > 1):
            ax1.patches.pop()
            ax1.patches.pop()

        patch1 = patches.FancyArrow(y_gt, x_gt, 0.5*np.cos(phi_gt), 0.5*np.sin(phi_gt), head_width=0.05, head_length=0.05, color='green')
        patch2 = patches.FancyArrow(y_pred, x_pred, 0.5*np.cos(phi_pred), 0.5*np.sin(phi_pred), head_width=0.05, head_length=0.05, color='blue')
        ax1.add_patch(patch1)
        ax1.add_patch(patch2)

        scatter2gt.set_offsets(np.array([-0.05, z_gt]))
        scatter2pr.set_offsets(np.array([0.05, z_pred]))

        #frame = frames[id].transpose(1, 2, 0)
        #frame = frame[:, :, 0]
        frame = frames[id].astype(np.uint8)
        imgplot.set_array(frame)

        annotation2.set_text('Frame {}'.format(id))

        #use the first one for viz on screen and second one for video recording
        #return plot1gt, plot1pr, patch1, patch2, scatter2gt, scatter2pr, imgplot, ax1, ax3, annotation, annotation2
        return plot1gt, plot1pr, patch1, patch2, scatter2gt, scatter2pr, imgplot, annotation, annotation2


    ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=1, blit=True)
    ani.save('himaxVsbebopExposure.avi', writer=writer)
    #ani.save('himaxVsbebop.gif', dpi=80, writer='imagemagick')
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

    model = FrontNet(PreActBlock, [1, 1, 1])
    ModelManager.Read('Models/FrontNetGrayExposure.pt', model)
    trainer = ModelTrainer(model)

    images = ImageIO.ReadImagesFromFolder("../data/himax_processed/", '.jpg', 0)
    [x_live, y_live] = DataProcessor.ProcessInferenceData(images, 60, 108)
    live_set = Dataset(x_live, y_live)
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 0}
    live_generator = data.DataLoader(live_set, **params)

    y_pred_himax = trainer.Infer(live_generator)
    y_pred_himax = np.reshape(y_pred_himax, (-1, 4))

    images = ImageIO.ReadImagesFromFolder("../data/bebop_processed/", '.jpg', 0)
    [x_live, y_live] = DataProcessor.ProcessInferenceData(images, 60, 108)
    live_set = Dataset(x_live, y_live)
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 0}
    live_generator = data.DataLoader(live_set, **params)

    y_pred_bebop = trainer.Infer(live_generator)
    y_pred_bebop = np.reshape(y_pred_bebop, (-1, 4))

    VizDroneBEV(images, y_pred_bebop, y_pred_himax)


if __name__ == '__main__':
    main()