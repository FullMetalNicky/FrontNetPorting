import torch.nn as nn
import torch
import numpy as np
from ValidationUtils import RunningAverage
from ValidationUtils import MovingAverage
from DataVisualization import DataVisualization
from EarlyStopping import EarlyStopping
from ValidationUtils import Metrics
import nemo
import logging
from ModelManager import ModelManager
from collections import OrderedDict

class ModelTrainer:
    def __init__(self, model, args, regime):
        self.num_epochs = args.epochs
        self.args = args
        self.model = model
        self.regime = regime
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        logging.info("[ModelTrainer] " + device)
        self.device = torch.device(device)
        self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.L1Loss()
        if self.args.quantize:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=float(regime['lr']), weight_decay=float(regime['weight_decay']))
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.relax = None
        self.folderPath = "Models/"

    def GetModel(self):
        return self.model

    def Quantize(self, validation_loader):

        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
             validation_loader)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        print("[ModelTrainer]: Before quantization process: %f" % acc)

        # [NeMO] This call "transforms" the model into a quantization-aware one, which is printed immediately afterwards.
        self.model = nemo.transform.quantize_pact(self.model)
        logging.info("[ModelTrainer] Model: %s", self.model)
        # [NeMO] NeMO re-training usually converges better using an Adam optimizer, and a smaller learning rate
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.regime['lr']),
        #                              weight_decay=float(self.regime['weight_decay']))

        # [NeMO] DNNs that do not employ batch normalization layers nor have clipped activations (e.g. ReLU6) require
        # an initial calibration to transfer to a quantization-aware version. This is used to calibrate the scaling
        # parameters of quantization-aware activation layers to the point defined by the maximum activation value
        # seen during a validation run. DNNs that employ BN or ReLU6 (or both) do not require this operation, as their
        # activations are already statistically bounded in terms of dynamic range.
        # logging.info("[ModelTrainer] Gather statistics for non batch-normed activations")
    
        # this is with original-Dronet non-fixed PreActBlock
        self.model.fold_bn({
             'conv':            'layer1.bn1',
             'layer1.conv1':    'layer1.bn2',
        }, {
             'layer1.bn1':      'layer1.shortcut',
        })
        self.model.fold_bn({
             'layer1.shortcut': 'layer2.bn1',
             'layer1.conv2':    'layer2.bn1',
             'layer2.conv1':    'layer2.bn2',
        }, {
             'layer2.bn1':      'layer2.shortcut',
        })
        self.model.fold_bn({
             'layer2.shortcut': 'layer3.bn1',
             'layer2.conv2':    'layer3.bn1',
             'layer3.conv1':    'layer3.bn2',
        }, {
             'layer3.bn1':      'layer3.shortcut',
        })

        # # this is with fixed PreActBlock
        # self.model.fold_bn({
        #     'conv':            'layer1.bn1',
        #     'layer1.conv1':    'layer1.bn2',
        #     'layer1.shortcut': 'layer2.bn1',
        #     'layer1.conv2':    'layer2.bn1',
        #     'layer2.conv1':    'layer2.bn2',
        #     'layer2.shortcut': 'layer3.bn1',
        #     'layer2.conv2':    'layer3.bn1',
        #     'layer3.conv1':    'layer3.bn2',
        # })


        self.model.reset_alpha_weights()
        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
             validation_loader)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        print("[ModelTrainer]: After BN folding: %f" % acc)


        self.model.set_statistics_act()
        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            validation_loader)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        self.model.unset_statistics_act()
        self.model.reset_alpha_act()
        logging.info("[ModelTrainer] statistics %.2f" % acc)

        precision_rule = self.regime['relaxation']

        # [NeMO] Change precision and reset weight clipping parameters
        self.model.change_precision(bits=16)
        self.model.reset_alpha_weights()

        # [NeMO] Export legacy-style INT-16 weights. Clipping parameters are changed!
        self.model.export_weights_legacy_int16(save_binary=True)
        # [NeMO] Re-check validation accuracy
        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
             validation_loader)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        print("[ModelTrainer]: After export: %f" % acc)


    def TrainSingleEpoch(self, training_generator):

        self.model.train()
        train_loss_x = MovingAverage()
        train_loss_y = MovingAverage()
        train_loss_z = MovingAverage()
        train_loss_phi = MovingAverage()

        i = 0
        for batch_samples, batch_targets in training_generator:

            batch_targets = batch_targets.to(self.device)
            batch_samples = batch_samples.to(self.device)
            outputs = self.model(batch_samples)

            loss_x = self.criterion(outputs[0], (batch_targets[:, 0]).view(-1, 1))
            loss_y = self.criterion(outputs[1], (batch_targets[:, 1]).view(-1, 1))
            loss_z = self.criterion(outputs[2], (batch_targets[:, 2]).view(-1, 1))
            loss_phi = self.criterion(outputs[3], (batch_targets[:, 3]).view(-1, 1))
            loss = loss_x + loss_y + loss_z + loss_phi

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss_x.update(loss_x)
            train_loss_y.update(loss_y)
            train_loss_z.update(loss_z)
            train_loss_phi.update(loss_phi)

            if (i + 1) % 100 == 0:
                logging.info("[ModelTrainer] Step [{}]: Average train loss {}, {}, {}, {}".format(i+1, train_loss_x.value, train_loss_y.value, train_loss_z.value,
                                                           train_loss_phi.value))
            i += 1

        return train_loss_x.value, train_loss_y.value, train_loss_z.value, train_loss_phi.value


    def ValidateSingleEpoch(self, validation_generator):

        self.model.eval()
        valid_loss = RunningAverage()
        valid_loss_x = RunningAverage()
        valid_loss_y = RunningAverage()
        valid_loss_z = RunningAverage()
        valid_loss_phi = RunningAverage()

        y_pred = []
        gt_labels = []
        with torch.no_grad():
            for batch_samples, batch_targets in validation_generator:
                gt_labels.extend(batch_targets.cpu().numpy())
                batch_targets = batch_targets.to(self.device)
                batch_samples = batch_samples.to(self.device)
                outputs = self.model(batch_samples)

                loss_x = self.criterion(outputs[0], (batch_targets[:, 0]).view(-1, 1))
                loss_y = self.criterion(outputs[1], (batch_targets[:, 1]).view(-1, 1))
                loss_z = self.criterion(outputs[2], (batch_targets[:, 2]).view(-1, 1))
                loss_phi = self.criterion(outputs[3], (batch_targets[:, 3]).view(-1, 1))
                loss = loss_x + loss_y + loss_z + loss_phi

                valid_loss.update(loss)
                valid_loss_x.update(loss_x)
                valid_loss_y.update(loss_y)
                valid_loss_z.update(loss_z)
                valid_loss_phi.update(loss_phi)

                outputs = torch.stack(outputs, 0)
                outputs = torch.squeeze(outputs)
                outputs = torch.t(outputs)
                y_pred.extend(outputs.cpu().numpy())

            logging.info("[ModelTrainer] Average validation loss {}, {}, {}, {}".format(valid_loss_x.value, valid_loss_y.value,
                                                                  valid_loss_z.value,
                                                                  valid_loss_phi.value))


        return valid_loss_x.value, valid_loss_y.value, valid_loss_z.value, valid_loss_phi.value, y_pred, gt_labels

    def Train(self, training_generator, validation_generator):

        metrics = Metrics()
        early_stopping = EarlyStopping(patience=10, verbose=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=np.sqrt(0.1),
                                                                    patience=5, verbose=False,
                                                                    threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                                    min_lr=0.1e-6, eps=1e-08)
        loss_epoch_m1 = 1e3

        for epoch in range(1, self.args.epochs + 1):
            logging.info("[ModelTrainer] Starting Epoch {}".format(epoch))

            change_prec = False
            ended = False
            # if self.args.quantize:
            #     change_prec, ended = self.relax.step(loss_epoch_m1, epoch, None)
            # if ended:
            #     break

            train_loss_x, train_loss_y, train_loss_z, train_loss_phi = self.TrainSingleEpoch(training_generator)

            valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
                validation_generator)

            valid_loss = valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi
            scheduler.step(valid_loss)

            gt_labels = torch.tensor(gt_labels, dtype=torch.float32)
            y_pred = torch.tensor(y_pred, dtype=torch.float32)
            MSE, MAE, r2_score = metrics.Update(y_pred, gt_labels,
                                               [train_loss_x, train_loss_y, train_loss_z, train_loss_phi],
                                               [valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi])

            logging.info('[ModelTrainer] Validation MSE: {}'.format(MSE))
            logging.info('[ModelTrainer] Validation MAE: {}'.format(MAE))
            logging.info('[ModelTrainer] Validation r2_score: {}'.format(r2_score))

            checkpoint_filename = self.folderPath + self.model.name + '-{:03d}.pt'.format(epoch)
            early_stopping(valid_loss, self.model, epoch, checkpoint_filename)
            if early_stopping.early_stop:
                logging.info("[ModelTrainer] Early stopping")
                break

        MSEs = metrics.GetMSE()
        MAEs = metrics.GetMAE()
        r2_score = metrics.Get()
        y_pred_viz = metrics.GetPred()
        gt_labels_viz = metrics.GetLabels()
        train_losses_x, train_losses_y, train_losses_z, train_losses_phi, valid_losses_x, valid_losses_y, valid_losses_z, valid_losses_phi = metrics.GetLosses()

        DataVisualization.desc = "Train_"
        DataVisualization.PlotLoss(train_losses_x, train_losses_y, train_losses_z, train_losses_phi , valid_losses_x, valid_losses_y, valid_losses_z, valid_losses_phi)
        DataVisualization.PlotMSE(MSEs)
        DataVisualization.PlotMAE(MAEs)
        DataVisualization.PlotR2Score(r2_score)

        DataVisualization.PlotGTandEstimationVsTime(gt_labels_viz, y_pred_viz)
        DataVisualization.PlotGTVsEstimation(gt_labels_viz, y_pred_viz)
        DataVisualization.DisplayPlots()

    def PerdictSingleSample(self, test_generator):

        iterator = iter(test_generator)
        batch_samples, batch_targets = iterator.next()
        index = np.random.choice(np.arange(0, batch_samples.shape[0]), 1)
        x_test = batch_samples[index]
        y_test = batch_targets[index]
        self.model.eval()

        logging.info('[ModelTrainer] GT Values: {}'.format(y_test.cpu().numpy()))
        with torch.no_grad():
            x_test = x_test.to(self.device)
            outputs = self.model(x_test)

        outputs = torch.stack(outputs, 0)
        outputs = torch.squeeze(outputs)
        outputs = torch.t(outputs)
        outputs = outputs.cpu().numpy()
        logging.info('[ModelTrainer] Prediction Values: {}'.format(outputs))
        return x_test[0].cpu().numpy(), y_test[0], outputs


    def InferSingleSample(self, frame):

        shape = frame.shape
        if len(frame.shape) == 3:
            frame = np.reshape(frame, (1, shape[0], shape[1], shape[2]))

        frame = np.swapaxes(frame, 1, 3)
        frame = np.swapaxes(frame, 2, 3)
        frame = frame.astype(np.float32)
        frame = torch.from_numpy(frame)
        self.model.eval()

        with torch.no_grad():
            frame = frame.to(self.device)
            outputs = self.model(frame)

        outputs = torch.stack(outputs, 0)
        outputs = torch.squeeze(outputs)
        #outputs = torch.t(outputs)
        #outputs = outputs.cpu().numpy()
        return outputs

    def Predict(self, test_generator):

        metrics = Metrics()

        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            test_generator)

        gt_labels = torch.tensor(gt_labels, dtype=torch.float32)
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
        MSE, MAE, r2_score = metrics.Update(y_pred, gt_labels,
                                           [0, 0, 0, 0],
                                           [valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi])

        y_pred_viz = metrics.GetPred()
        gt_labels_viz = metrics.GetLabels()

        DataVisualization.desc = "Test_"
        DataVisualization.PlotGTandEstimationVsTime(gt_labels_viz, y_pred_viz)
        DataVisualization.PlotGTVsEstimation(gt_labels_viz, y_pred_viz)
        DataVisualization.DisplayPlots()
        logging.info('[ModelTrainer] Test MSE: {}'.format(MSE))
        logging.info('[ModelTrainer] Test MAE: {}'.format(MAE))
        logging.info('[ModelTrainer] Test r2_score: {}'.format(r2_score))

    def Infer(self, live_generator):

        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            live_generator)

        return y_pred

