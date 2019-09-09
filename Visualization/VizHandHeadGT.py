from __future__ import print_function

import logging
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

import sys
sys.path.append("../PyTorch/")
from DataProcessor import DataProcessor
from Dataset import Dataset


def VizDroneBEV(frames, hand_labels, head_labels):

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
    ax1.invert_xaxis()
    trianglex = [2, 0, -2, 2]
    triangley = [3, 0, 3, 3]
    collection = plt.fill(trianglex, triangley, facecolor='lightskyblue')

    plot1gthead, = plt.plot([], [], color='green', label='Head GT', linestyle='None', marker='o', markersize=10)
    arr1gthead = ax1.arrow([], [], np.cos([]), np.sin([]), head_width=0.1, head_length=0.1, color='green',
                           animated=True)

    plot1gthand, = plt.plot([], [], color='blue', label='Hand GT', linestyle='None', marker='o', markersize=10)
    arr1gthand = ax1.arrow([], [], np.cos([]), np.sin([]), head_width=0.1, head_length=0.1, color='blue',
                           animated=True)

    plt.legend(loc='lower right', bbox_to_anchor=(0.8, 0.2, 0.25, 0.25))

    ax2 = plt.subplot2grid((h, w), (2, 8), rowspan=7)
    ax2.set_title('Relative z', pad=20)
    ax2.yaxis.tick_right()
    ax2.set_xlim([-0.5, 0.5])
    ax2.set_xticklabels([])
    ax2.yaxis.set_ticks([-1, 0, 1])  # set y-ticks
    ax2.xaxis.set_ticks_position('none')
    scatter2gthead = plt.scatter([], [], color='green', label='Head GT', s=100)
    scatter2gthand = plt.scatter([], [], color='blue', label='Hand GT', s=100)

    ax3 = plt.subplot2grid((h, w), (2, 9), rowspan=7, colspan=7)
    ax3.axis('off')
    frame = frames[0].transpose(1, 2, 0)
    frame = frame.astype(np.uint8)
    imgplot = plt.imshow(frame)

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
    writer = Writer(fps=20, metadata=dict(artist='FullMetalNicky'))


    def animate(id):
        head_label = head_labels[id]
        hand_label = hand_labels[id]

        x_head, y_head, z_head, phi_head = head_label[0], head_label[1], head_label[2], head_label[3]
        x_hand, y_hand, z_hand, phi_hand = hand_label[0], hand_label[1], hand_label[2], hand_label[3]

        str1 = "x_head={:05.3f}, y_head={:05.3f}, z_head={:05.3f}, phi_head={:05.3f} {}".format(x_head, y_head, z_head,
                                                                                                phi_head, "\n")
        str1 = str1 + "x_hand={:05.3f}, y_hand={:05.3f}, z_hand={:05.3f}, phi_hand={:05.3f} {}".format(x_hand, y_hand,
                                                                                                       z_hand, phi_hand,
                                                                                                       "\n")

        phi_head = - phi_head - np.pi/2
        phi_hand = - phi_hand - np.pi/2

        annotation.set_text(str1)

        plot1gthead.set_data(np.array([y_head, x_head]))
        plot1gthand.set_data(np.array([y_hand, x_hand]))

        if(len(ax1.patches) > 1):
            ax1.patches.pop()
            ax1.patches.pop()

        patch1 = patches.FancyArrow(y_head, x_head, 0.5*np.cos(phi_head), 0.5*np.sin(phi_head), head_width=0.05, head_length=0.05, color='green')
        patch2 = patches.FancyArrow(y_hand, x_hand, 0.5*np.cos(phi_hand), 0.5*np.sin(phi_hand), head_width=0.05, head_length=0.05, color='blue')
        ax1.add_patch(patch1)
        ax1.add_patch(patch2)

        scatter2gthead.set_offsets(np.array([-0.05, z_head]))
        scatter2gthand.set_offsets(np.array([-0.0, z_hand]))

        frame = frames[id].transpose(1, 2, 0)
        frame = frame.astype(np.uint8)
        imgplot.set_array(frame)

        annotation2.set_text('Frame {}'.format(id))

        #use the first one for viz on screen and second one for video recording
        #return plot1gt, plot1pr, patch1, patch2, scatter2gt, scatter2pr, imgplot, ax1, ax3, annotation, annotation2
        return plot1gthead, plot1gthand, patch1, patch2, scatter2gthead, scatter2gthand, imgplot, annotation, annotation2


    ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=1, blit=True)
    ani.save('Davide.mp4', writer=writer)
    #ani.save('viz2.gif', dpi=80, writer='imagemagick')
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


    DATA_PATH = "/Users/usi/PycharmProjects/data/"

    [x_test, y_test, z_test] = DataProcessor.ProcessTestData(DATA_PATH + "DavideHand.pickle", 60, 108)

    test_set = Dataset(x_test, y_test)
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 0}


    VizDroneBEV(x_test, y_test, z_test)


if __name__ == '__main__':
    main()