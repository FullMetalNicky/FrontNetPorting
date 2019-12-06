import torch
import sklearn.metrics
import numpy as np

class AverageBase(object):

    def __init__(self, value=0):
        self.value = float(value) if value is not None else None

    def __str__(self):
        return str(round(self.value, 4))

    def __repr__(self):
        return self.value

    def __format__(self, fmt):
        return self.value.__format__(fmt)

    def __float__(self):
        return self.value


class RunningAverage(AverageBase):
    """
    Keeps track of a cumulative moving average (CMA).
    """

    def __init__(self, value=0, count=0):
        super(RunningAverage, self).__init__(value)
        self.count = count

    def update(self, value):
        self.value = (self.value * self.count + float(value))
        self.count += 1
        self.value /= self.count
        return self.value


class MovingAverage(AverageBase):
    """
    An exponentially decaying moving average (EMA).
    """

    def __init__(self, alpha=0.99):
        super(MovingAverage, self).__init__(None)
        self.alpha = alpha

    def update(self, value):
        if self.value is None:
            self.value = float(value)
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * float(value)
        return self.value


class Metrics:
    def __init__(self):
        self.MSE = []
        self.MAE = []
        self.r2_score = []

        self.train_losses_x = []
        self.train_losses_y = []
        self.train_losses_z = []
        self.train_losses_phi = []
        self.valid_losses_x = []
        self.valid_losses_y = []
        self.valid_losses_z = []
        self.valid_losses_phi = []
        self.gt_labels = []
        self.y_pred = []

    def Update(self, y_pred, gt_labels, train_loss, valid_loss):

        self.train_losses_x.append(train_loss[0])
        self.train_losses_y.append(train_loss[1])
        self.train_losses_z.append(train_loss[2])
        self.train_losses_phi.append(train_loss[3])

        self.valid_losses_x.append(valid_loss[0])
        self.valid_losses_y.append(valid_loss[1])
        self.valid_losses_z.append(valid_loss[2])
        self.valid_losses_phi.append(valid_loss[3])

        self.y_pred.append(y_pred)
        self.gt_labels.append(gt_labels)

        MSE = torch.mean((y_pred - gt_labels).pow(2), 0)
        MAE = torch.mean(torch.abs(y_pred - gt_labels), 0)

        x = y_pred[:, 0]
        x = x.cpu().numpy()
        x_gt = gt_labels[:, 0]
        x_gt = x_gt.cpu().numpy()

        y = y_pred[:, 1]
        y = y.cpu().numpy()
        y_gt = gt_labels[:, 1]
        y_gt = y_gt.cpu().numpy()

        z = y_pred[:, 2]
        z = z.cpu().numpy()
        z_gt = gt_labels[:, 2]
        z_gt = z_gt.cpu().numpy()

        phi = y_pred[:, 3]
        phi = phi.cpu().numpy()
        phi_gt = gt_labels[:, 3]
        phi_gt = phi_gt.cpu().numpy()

        x_r2 = sklearn.metrics.r2_score(x_gt, x)
        y_r2 = sklearn.metrics.r2_score(y_gt, y)
        z_r2 = sklearn.metrics.r2_score(z_gt, z)
        phi_r2 = sklearn.metrics.r2_score(phi_gt, phi)
        r2_score = torch.FloatTensor([x_r2, y_r2, z_r2, phi_r2])

        self.MSE.append(MSE)
        self.MAE.append(MAE)
        self.r2_score.append(r2_score)

        return MSE, MAE, r2_score

    def Reset(self):
        self.MSE.clear()
        self.MAE.clear()
        self.r2_score.clear()

    def GetPred(self):
        return self.y_pred

    def GetLabels(self):
        return self.gt_labels

    def GetLosses(self):
        return self.train_losses_x, self.train_losses_y, self.train_losses_z, self.train_losses_phi , \
               self.valid_losses_x, self.valid_losses_y, self.valid_losses_z, self.valid_losses_phi

    def GetMSE(self):
        return self.MSE

    def GetMAE(self):
        return self.MAE

    def Get(self):
        return self.r2_score
