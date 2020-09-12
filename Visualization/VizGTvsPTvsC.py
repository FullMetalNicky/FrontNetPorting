from __future__ import print_function
import logging
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
import matplotlib.patches as patches
import pandas as pd


sys.path.append("../PyTorch/")


def PlotGTVsEstimationScatter(gt, c_pred):
    fig = plt.figure(666, figsize=(16, 6))
    mark_size = 1

    gs = gridspec.GridSpec(1, 3)
    ax = plt.subplot(gs[0, 0], xlim=(0.5, 2.6), ylim=(0.5, 2.6))
    ax.set_title('Output variable: x', fontsize=18)
    ax.set_xlabel('Ground Truth', fontsize=18)
    ax.set_ylabel('Quantized model prediction', fontsize=18)
    ax.set_xmargin(0.2)
    ax.scatter(c_pred[:, 0], gt[:, 0], color='g', marker='*', s=mark_size)
    ax.plot(gt[:, 0], gt[:, 0], color='black')


    ax = plt.subplot(gs[0, 1], xlim=(-1, 1.2), ylim=(-1, 1.2))
    ax.set_title('Output variable: y', fontsize=18)
    ax.set_xlabel('Ground Truth', fontsize=18)
    ax.set_ylabel('Quantized model prediction', fontsize=18)
    ax.set_xmargin(0.2)
    ax.scatter(c_pred[:, 1], gt[:, 1], color='blue', marker='*', s=mark_size)
    ax.plot(gt[:, 1], gt[:, 1], color='black')

    ax = plt.subplot(gs[0, 2], xlim=(-np.pi*0.6, np.pi*0.6), ylim=(-np.pi*0.6, np.pi*0.6))
    ax.set_title('Output variable: phi', fontsize=18)
    ax.set_xlabel('Ground Truth', fontsize=18)
    ax.set_ylabel('Quantized model prediction', fontsize=18)
    ax.set_xmargin(0.2)
    ax.scatter(c_pred[:, 3], 0.377 + gt[:, 3], color='orangered', marker='*', s=mark_size)
    ax.plot(0.377 + gt[:, 3], 0.377 + gt[:, 3], color='black')


    plt.suptitle('Final Evaluation - Ground Truth vs Predictions', fontsize=22)
    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    plt.savefig('FinalEvalPRedVdGTScatter.png')
    #plt.subplots_adjust(hspace=0.2)
    #plt.subplots_adjust(top=0.85)
    #plt.tight_layout()

    plt.show()




def PlotGTVsEstimation(gt, c_pred, py_pred):
    fig = plt.figure(666, figsize=(16, 16))
    length = len(gt)
    mark_size = 1

    gs = gridspec.GridSpec(3, 1)
    ax = plt.subplot(gs[0, 0], xlim=(-10,length+300))
    ax.set_title('Output variable: x', fontsize=18)
    ax.set_xlabel('Frame number', fontsize=18)
    ax.set_ylabel('x', fontsize=18)
    ax.set_xmargin(0.2)
    ax.plot(range(0, length), gt[:, 0], color='g', marker='*', label="Ground Truth", markersize=mark_size)
    ax.plot(range(0, length), c_pred[:, 0], color='r', marker='D', label="Quantized model prediction", markersize=mark_size)
   # ax.plot(range(0, length), py_pred[:, 0], color='b', marker='o', label="FP", markersize=mark_size)

    plt.legend(fontsize=15)

    ax = plt.subplot(gs[1, 0], xlim=(-10,length+300))
    ax.set_title('Output variable: y', fontsize=18)
    ax.set_xlabel('Frame number', fontsize=18)
    ax.set_ylabel('y', fontsize=18)
    ax.set_xmargin(0.2)
    ax.plot(range(0, length), gt[:, 1], color='g', marker='*', label="Ground Truth", markersize=mark_size)
    ax.plot(range(0, length), c_pred[:, 1], color='r', marker='D', label="Quantized model prediction", markersize=mark_size)
   # ax.plot(range(0, length), py_pred[:, 1], color='b', marker='o', label="FP", markersize=mark_size)
    plt.legend(fontsize=15)

    ax = plt.subplot(gs[2,0], xlim=(-10,length+300))
    ax.set_title('Output variable: phi', fontsize=18)
    ax.set_xlabel('Frame number', fontsize=18)
    ax.set_ylabel('phi', fontsize=18)
    ax.set_xmargin(0.2)
    ax.plot(range(0, length), 0.377+gt[:, 3], color='g', marker='*', label="Ground Truth", markersize=mark_size)
    ax.plot(range(0, length), c_pred[:, 3], color='r', marker='D', label="Quantized model prediction", markersize=mark_size)
  #  ax.plot(range(0, length), py_pred[:, 3], color='b', marker='o', label="FP", markersize=mark_size)
    plt.legend(fontsize=15)



    plt.suptitle('Final Evaluation - Ground Truth vs Predictions', fontsize=22)
    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
    plt.savefig('FinalEvalPRedVdGT.png')

    plt.show()

def ErrorHistograms(gt, c_pred):
    plt.figure(666, figsize=(16, 16))
    length = len(gt)



    gs = gridspec.GridSpec(3, 1)
    ax = plt.subplot(gs[0, 0], xlim=(-0.75,0.75))
    ax.set_title('x')
    ax.set_xlabel('x')
    ax.set_ylabel('error')
    ax.set_xmargin(0.2)
    n, bins, patches = plt.hist(gt[:, 0] - c_pred[:,0], bins=np.arange(-0.75, 0.75, 0.07), density=True, facecolor='g', alpha=0.75, edgecolor='black')


    ax = plt.subplot(gs[1, 0],  xlim=(-2,2))
    ax.set_title('y')
    ax.set_xlabel('y')
    ax.set_ylabel('error')
    ax.set_xmargin(0.2)
    n, bins, patches = plt.hist(gt[:, 1]- c_pred[:,1], bins=np.arange(-2, 2, 4/23), density=True, facecolor='g', alpha=0.75, edgecolor='black')
    print(-np.sort(-gt[:, 1] - c_pred[:, 1]))

    ax = plt.subplot(gs[2, 0],  xlim=(-1.5,1.5))
    ax.set_title('phi')
    ax.set_xlabel('phi')
    ax.set_ylabel('error')
    ax.set_xmargin(0.2)
    n, bins, patches = plt.hist(0.377 + gt[:, 3]- c_pred[:,3], bins=np.arange(-1.5, 1.5, 0.1), density=True, facecolor='g', alpha=0.75, edgecolor='black')


    plt.subplots_adjust(hspace=0.3)
    plt.suptitle('Final Evaluation - Error Histogram')
    plt.savefig('FinalEvalErrorHist.png')

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

    picklename = "pickles/first_outputs.pickle"
    test_set = pd.read_pickle(picklename)

    gt = test_set['y'].values
    gt = np.vstack(gt[:]).astype(np.float32)
    c_pred = test_set['o'].values
    c_pred = np.vstack(c_pred[:]).astype(np.float32)
    py_pred = test_set['z'].values
    py_pred = np.vstack(py_pred[:]).astype(np.float32)

    #PlotGTVsEstimation(gt[100:1100], c_pred[100:1100], py_pred[100:1100])
    #PlotGTVsEstimation(gt[200:1500], c_pred[200:1500], py_pred[200:1500])
    #ErrorHistograms(gt[200:1500], c_pred[200:1500])
    PlotGTVsEstimationScatter(gt[200:1500], c_pred[200:1500])



if __name__ == '__main__':
    main()