from __future__ import print_function
import logging
import numpy as np
import pandas as pd
import sys
import torch



def SaveModelResultsToCSV(MSE, MAE, r2_score, labels, predictions, name):
    x_gt = []
    y_gt = []
    z_gt = []
    phi_gt = []
    x_pr = []
    y_pr = []
    z_pr = []
    phi_pr = []

    r2_score = torch.stack(r2_score, 0)
    x = r2_score[:, 0]
    r2_score_x = x.cpu().numpy()
    y = r2_score[:, 1]
    r2_score_y = y.cpu().numpy()
    z = r2_score[:, 2]
    r2_score_z = z.cpu().numpy()
    phi = r2_score[:, 3]
    r2_score_phi = phi.cpu().numpy()

    MSE = torch.stack(MSE, 0)
    x = MSE[:, 0]
    MSE_x = x.cpu().numpy()
    y = MSE[:, 1]
    MSE_y = y.cpu().numpy()
    z = MSE[:, 2]
    MSE_z = z.cpu().numpy()
    phi = MSE[:, 3]
    MSE_phi = phi.cpu().numpy()

    MAE = torch.stack(MAE, 0)
    x = MAE[:, 0]
    MAE_x = x.cpu().numpy()
    y = MAE[:, 1]
    MAE_y = y.cpu().numpy()
    z = MAE[:, 2]
    MAE_z = z.cpu().numpy()
    phi = MAE[:, 3]
    MAE_phi = phi.cpu().numpy()

    # predictions = torch.stack(predictions, 0)
    # predictions = np.reshape(predictions, (-1, 4))
    # x = predictions[:, 0]
    # predictions_x = x.cpu().numpy()
    # y = predictions[:, 1]
    # predictions_y = y.cpu().numpy()
    # z = predictions[:, 2]
    # predictions_z = z.cpu().numpy()
    # phi = predictions[:, 3]
    # predictions_phi = phi.cpu().numpy()
    #
    # labels = torch.stack(labels, 0)
    # labels = np.reshape(labels, (-1, 4))
    # x = labels[:, 0]
    # labels_x = x.cpu().numpy()
    # y = labels[:, 1]
    # labels_y = y.cpu().numpy()
    # z = labels[:, 2]
    # labels_z = z.cpu().numpy()
    # phi = labels[:, 3]
    # labels_phi = phi.cpu().numpy()





    df = pd.DataFrame(
        data={ 'MSE_x': MSE_x, 'MSE_y': MSE_y, 'MSE_z': MSE_z, 'MSE_phi': MSE_phi,
              'MAE_x': MAE_x, 'MAE_y': MAE_y, 'MAE_z': MAE_z, 'MAE_phi': MAE_phi,
               'r2_score_x': r2_score_x, 'r2_score_y': r2_score_y, 'r2_score_z': r2_score_z, 'r2_score_phi': r2_score_phi})
    df.index.name = "epochs"

    df.to_csv(name + ".csv", header=True)


