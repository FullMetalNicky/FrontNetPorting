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
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.animation as animation
import matplotlib.pyplot as plt




# def VizOneFrameAtATime(trainer, test_generator, viz_function):
#
#     for batch_samples, batch_targets in test_generator:
#         outputs = trainer.PerdictSingleSample(batch_samples)
#         #fig = viz_function(batch_samples[0].cpu().numpy(), batch_targets[0], outputs)
#         # canvas = FigureCanvas(fig)
#         # canvas.draw()
#         #
#         # img = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8, sep='')
#         # img = img.reshape(canvas.get_width_height()[::-1] + (3,))
#         # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         #
#         # cv2.imshow("plot", img)
#         # cv2.waitKey(5)
#         DataVisualization.DisplayVideoFrame(batch_samples[0].cpu().numpy())



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
x_test = x_test[:1000]
y_test = y_test[:1000]
test_set = Dataset(x_test, y_test)
params = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 0}
test_generator = data.DataLoader(test_set, **params)

trainer = ModelTrainer(model)

frames = x_test
labels = y_test
valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, outputs, gt_labels = trainer.ValidateSingleEpoch(test_generator)

fig = plt.figure(666, figsize=(10, 6))

w = 20
h = 12
bar_length = h - 2
offset_x = int((w - bar_length) / 2)
ax1 = plt.subplot2grid((h, w), (0, offset_x), colspan=bar_length)
ax1.set_title('x')
ax1.xaxis.tick_top()
ax1.set_xlim([0, 4])
ax1.set_ylim([-0.5, 0.5])
ax1.set_yticklabels([])
scatter1gt = plt.scatter([], [], color='green', label='GT', s=100)
scatter1pr = plt.scatter([], [], color='blue', label='Prediction', s=100)

ax2 = plt.subplot2grid((h, w), (1, 0), rowspan=bar_length)
ax2.set_title('y')
ax2.set_ylim([-1, 1])
ax2.set_xlim([-0.5, 0.5])
ax2.set_xticklabels([])
scatter2gt = plt.scatter([], [], color='green', label='GT', s=100)
scatter2pr = plt.scatter([], [], color='blue', label='Prediction', s=100)

ax3 = plt.subplot2grid((h, w), (1, 1), rowspan=bar_length, colspan=(w - 2))
ax3.set_yticklabels([])
ax3.set_xticklabels([])
frame = frames[0].transpose(1, 2, 0)
frame = frame.astype(np.uint8)
imgplot = plt.imshow(frame)

ax4 = plt.subplot2grid((h, w), (1, w-1), rowspan=bar_length)
ax4.set_title('z')
ax4.yaxis.tick_right()
ax4.set_ylim([-1, 1])
ax4.set_xlim([-0.5, 0.5])
ax4.set_xticklabels([])
scatter3gt = plt.scatter([], [], color='green', label='GT', s=100)
scatter3pr = plt.scatter([], [], color='blue', label='Prediction', s=100)

ax5 = plt.subplot2grid((h, w), (h-1, offset_x), colspan=bar_length)
ax5.set_title('phi')
ax5.set_xlim([-2, 2])
ax5.set_ylim([-0.5, 0.5])
ax5.set_yticklabels([])
scatter4gt = plt.scatter([], [], color='green', label='GT', s=100)
scatter4pr = plt.scatter([], [],  color='blue', label='Prediction', s=100)

plt.subplots_adjust(hspace=1.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'))


def animate(id):
    label = labels[id]
    scatter1gt.set_offsets(np.array([label[0], -0.05]))
    scatter1pr.set_offsets(np.array([outputs[id*4 + 0], 0.05]))

    scatter2gt.set_offsets(np.array([-0.05, label[1]]))
    scatter2pr.set_offsets(np.array([0.05, outputs[id*4 + 1]]))

    frame = frames[id].transpose(1, 2, 0)
    frame = frame.astype(np.uint8)
    imgplot.set_array(frame)

    scatter3gt.set_offsets(np.array([-0.05, label[2]]))
    scatter3pr.set_offsets(np.array([0.05, outputs[id * 4 + 2]]))

    scatter4gt.set_offsets(np.array([label[3], -0.05]))
    scatter4pr.set_offsets(np.array([outputs[id * 4 + 3], 0.05]))

    return scatter1gt, scatter1pr, scatter2gt, scatter2pr, imgplot, scatter3gt, scatter3pr, scatter4gt, scatter4pr


ani = animation.FuncAnimation(fig, animate, frames=len(x_test), interval=1, blit=True)
ani.save('plot.avi', writer=writer)
plt.show()

