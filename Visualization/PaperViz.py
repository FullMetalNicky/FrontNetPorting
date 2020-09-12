import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
import matplotlib.patches as patches

import sys
sys.path.append("../PyTorch/")
from DataProcessor import DataProcessor


def DrawYaw(y_test):
    yaw = y_test[:, 3]
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(range(len(y_test)), yaw, '.-')
    ax.set(xlabel='frames', ylabel='yaw',
           title='yaw vs time')
    ax.grid()
    fig.savefig("yawvstime.png")
    plt.show()

def GetDesiredLocation(head_pose):

    dist = 1.3
    x, y, yaw = head_pose[0], head_pose[1], head_pose[3]

    return x + dist * np.cos(yaw), y + dist * np.sin(yaw), yaw + np.pi



def VizWorldTopView(frames, drone_poses, head_poses, desired_drone_poses, isGray=False, name="WorldTopViewPatterns"):

    fig = plt.figure(888, figsize=(15, 8))

    h = 9
    w = 16

    ax0 = plt.subplot2grid((h, w), (0, 0), colspan=7, rowspan=2)
    ax0.set_ylim([0, 9])
    annotation = ax0.annotate("poop", xy=(0, 5.5), size=10)
    annotation.set_animated(True)
    ax0.axis('off')

    ax1 = plt.subplot2grid((h, w), (2, 0), colspan=7, rowspan=7)
    ax1.set_title('World Frame Pose')
    plt.xlim(3.0, -3.0)
    plt.ylim(-3.0, 3.0)

    ax1.xaxis.tick_top()  # and move the X-Axis
    ax1.yaxis.tick_left()  # remove right y-Ticks
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_xlabel('Y')
    ax1.set_ylabel('X')

    plot1gt, = plt.plot([], [], color='green', label='Head', linestyle='None', marker='o', markersize=10)
    plot1pr, = plt.plot([], [], color='blue', label='Desired drone pose', linestyle='None', marker='^', markersize=10)
    plot1cam, = plt.plot([], [], color='k', label='Drone pose', linestyle='None', marker='s', markersize=10)
    arr1gt = ax1.arrow([], [], np.cos([]), np.sin([]), head_width=0.1, head_length=0.1, color='green', animated=True)
    arr1pr = ax1.arrow([], [], np.cos([]), np.sin([]), head_width=0.1, head_length=0.1, color='blue', animated=True)
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
    writer = Writer(fps=17, metadata=dict(artist='FullMetalNicky'))

    def animate(id):

        head_pose = head_poses[id]
        drone_pose = drone_poses[id]
        desired_drone_pose = desired_drone_poses[id]

        x_hd, y_hd, z_hd, phi_hd = head_pose[0], head_pose[1], head_pose[2], head_pose[3]
        x_cam, y_cam, z_cam, phi_cam = drone_pose[0], drone_pose[1], drone_pose[2], drone_pose[3]
        x_ds, y_ds, z_ds, phi_ds = desired_drone_pose[0], desired_drone_pose[1], desired_drone_pose[2], desired_drone_pose[3]

        str1 = "x_cam={:05.3f}, y_cam={:05.3f}, phi_cam={:05.3f} {}".format(x_cam, y_cam, phi_cam, "\n")
        str1 = str1 + "x_hd={:05.3f}, y_hd={:05.3f}, phi_hd={:05.3f} {}".format(x_hd, y_hd, phi_hd, "\n")
        str1 = str1 + "x_ds={:05.3f}, y_ds={:05.3f},  phi_ds={:05.3f}".format(x_ds, y_ds, phi_ds)

        annotation.set_text(str1)

        plot1gt.set_data(np.array([y_hd, x_hd]))
        plot1pr.set_data(np.array([y_ds, x_ds]))
        plot1cam.set_data(np.array([y_cam, x_cam]))

        if (len(ax1.patches) > 1):
            ax1.patches.pop()
            ax1.patches.pop()
            ax1.patches.pop()

        patch1 = patches.FancyArrow(y_hd, x_hd, 0.5 * np.sin(phi_hd), 0.5 * np.cos(phi_hd), head_width=0.05,
                                    head_length=0.05, color='green')
        patch2 = patches.FancyArrow(y_ds, x_ds, 0.5 * np.sin(phi_ds), 0.5 * np.cos(phi_ds), head_width=0.05,
                                    head_length=0.05, color='blue')
        patch3 = patches.FancyArrow(y_cam, x_cam, 0.5 * np.sin(phi_cam), 0.5 * np.cos(phi_cam), head_width=0.05,
                                    head_length=0.05, color='k')

        ax1.add_patch(patch1)
        ax1.add_patch(patch2)
        ax1.add_patch(patch3)

        frame = frames[id].astype(np.uint8)
        if isGray == False:
            frame = frame.transpose(1, 2, 0)
        imgplot.set_array(frame)

        annotation2.set_text('Frame {}'.format(id))

        # use the first one for viz on screen and second one for video recording
        # return plot1gt, plot1pr, patch1, patch2, scatter2gt, scatter2pr, imgplot, ax1, ax3, annotation, annotation2
        #return plot1gt, plot1pr, patch1, patch2, scatter2gt, scatter2pr, imgplot, annotation, annotation2

        return plot1gt, plot1pr, plot1cam, patch1, patch2, patch3, imgplot, annotation, annotation2

    ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=1, blit=True)
    ani.save(name + '.mp4', writer=writer)
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


    DATA_PATH = "/Users/usi/PycharmProjects/data/160x160/"
    name = "daniele_walk2.pickle"

    [x_test, y_test, drone_poses] = DataProcessor.ProcessTestData(DATA_PATH + name, True)
    h = x_test.shape[2]
    w = x_test.shape[3]
    x_test = np.reshape(x_test, (-1, h, w))

    test_set = pd.read_pickle(DATA_PATH + name)
    head_poses = test_set['v'].values
    head_poses = np.vstack(head_poses[:]).astype(np.float32)

    desired_drone_poses =[]

    for i in range(len(y_test)):
        x, y, yaw = GetDesiredLocation(head_poses[i])
        desired_drone_poses.append([x, y, 0, yaw])


    VizWorldTopView(x_test, drone_poses, head_poses, desired_drone_poses, isGray=True, name="daniele_walk2withDesiredPose")


if __name__ == '__main__':
    main()