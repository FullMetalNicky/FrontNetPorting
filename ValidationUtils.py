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
        self.r_score = []

    def Update(self, y_pred, gt_labels):
        MSE = torch.sqrt(torch.mean((y_pred - gt_labels).pow(2), 0))
        MAE = torch.sqrt(torch.mean(torch.abs(y_pred - gt_labels), 0))

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
        r_score = torch.FloatTensor([x_r2, y_r2, z_r2, phi_r2])

        self.MSE.append(MSE)
        self.MAE.append(MAE)
        self.r_score.append(r_score)

        return MSE, MAE, r_score

    def Reset(self):
        self.MSE.clear()
        self.MAE.clear()
        self.r_score.clear()

    def GetMSE(self):
        return self.MSE

    def GetMAE(self):
        return self.MAE

    def Getr2_score(self):
        return self.r_score
