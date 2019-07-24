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

import matplotlib.patches as patches

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
ModelManager.Read('Models/FrontNet-097.pkl', model)

DATA_PATH = "/Users/usi/PycharmProjects/data/"
[x_test, y_test] = DataProcessor.ProcessTestData(DATA_PATH + "test.pickle", 60, 108)
#x_test = x_test[:100]
#y_test = y_test[:100]
test_set = Dataset(x_test, y_test)
params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 0}
test_generator = data.DataLoader(test_set, **params)
trainer = ModelTrainer(model)
frames = x_test
labels = y_test
valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, outputs, gt_labels = trainer.ValidateSingleEpoch(test_generator)

fig = plt.figure(888, figsize=(15, 5))

h = 6
w = 15

ax0 = plt.subplot2grid((h, w), (0, 0), colspan=7)
ax0.set_ylim([0, 8])
annotation = ax0.annotate("poop", xy=(0, 4), size=10)
annotation.set_animated(True)
ax0.axis('off')

ax1 = plt.subplot2grid((h, w), (2, 0), colspan=7, rowspan=4)
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

plot1gt, = plt.plot([], [], color='green', label='GT', linestyle='None', marker='o', markersize=10)
plot1pr, = plt.plot([], [], color='blue', label='Prediction', linestyle='None', marker='o', markersize=10)
arr1gt = ax1.arrow([], [], np.cos([]), np.sin([]), head_width=0.05, head_length=0.05, color='green', animated=True)
arr1pr = ax1.arrow([], [], np.cos([]), np.sin([]), head_width=0.05, head_length=0.05, color='blue', animated=True)
plt.legend(loc='lower right', bbox_to_anchor=(0.8, 0.2, 0.25, 0.25))

ax2 = plt.subplot2grid((h, w), (2, 7), rowspan=4)
ax2.set_title('Relative z', pad=20)
ax2.yaxis.tick_right()
ax2.set_xlim([-0.5, 0.5])
ax2.set_xticklabels([])
ax2.yaxis.set_ticks([-1, 0, 1])  # set y-ticks
ax2.xaxis.set_ticks_position('none')
scatter2gt = plt.scatter([], [], color='green', label='GT', s=100)
scatter2pr = plt.scatter([], [], color='blue', label='Prediction', s=100)

ax3 = plt.subplot2grid((h, w), (2, 8), rowspan=4, colspan=7)
ax3.axis('off')
frame = frames[0].transpose(1, 2, 0)
frame = frame.astype(np.uint8)
imgplot = plt.imshow(frame)

ax4 = plt.subplot2grid((h, w), (0, 8), colspan=7)
ax4.set_xlim([0, 8])
annotation2 = ax4.annotate("poop", xy=(3, 0), size=14, weight='bold')
annotation2.set_animated(True)
ax4.axis('off')

plt.subplots_adjust(wspace=1.5)

img = mpimg.imread('minidrone.jpg')
newax = fig.add_axes([0.248, 0.0, 0.1, 0.1], anchor='S')
newax.imshow(img)
newax.axis('off')

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'))

print(len(ax1.patches))

def animate(id):
    label = labels[id]
    x_gt = label[0]
    x_pred = outputs[id*4 + 0]
    y_gt = label[1]
    y_pred = outputs[id*4 + 1]
    z_gt = label[2]
    z_pred = outputs[id*4 + 2]
    phi_gt = label[3] - np.pi / 2
    phi_pred = outputs[id*4 + 3] - np.pi / 2

    str1 = "x_gt={:05.3f}, y_gt={:05.3f}, z_gt={:05.3f}, phi_gt={:05.3f} {}".format(x_gt, y_gt, z_gt, phi_gt, "\n")
    str1 = str1 + "x_pr={:05.3f}, y_pr={:05.3f}, z_pr={:05.3f}, phi_pr={:05.3f}".format(x_pred, y_pred, z_pred, phi_pred)

    annotation.set_text(str1)

    plot1gt.set_data(np.array([y_gt, x_gt]))
    plot1pr.set_data(np.array([y_pred, x_pred]))

    if(len(ax1.patches) > 1):
        ax1.patches.pop()
        ax1.patches.pop()

    patch1 = patches.FancyArrow(y_gt, x_gt, np.cos(phi_gt), np.sin(phi_gt), head_width=0.05, head_length=0.05, color='green')
    patch2 = patches.FancyArrow(y_pred, x_pred, np.cos(phi_pred), np.sin(phi_pred), head_width=0.05, head_length=0.05, color='blue')
    ax1.add_patch(patch1)
    ax1.add_patch(patch2)

    scatter2gt.set_offsets(np.array([-0.05, label[2]]))
    scatter2pr.set_offsets(np.array([0.05, outputs[id*4 + 2]]))

    frame = frames[id].transpose(1, 2, 0)
    frame = frame.astype(np.uint8)
    imgplot.set_array(frame)

    annotation2.set_text('Frame {}'.format(id))

    #use the first one for viz on screen and second one for video recording
    #return plot1gt, plot1pr, patch1, patch2, scatter2gt, scatter2pr, imgplot, ax1, ax3, annotation, annotation2
    return plot1gt, plot1pr, patch1, patch2, scatter2gt, scatter2pr, imgplot, annotation, annotation2


ani = animation.FuncAnimation(fig, animate, frames=len(x_test), interval=1, blit=True)
ani.save('droneplot.avi', writer=writer)
plt.show()
