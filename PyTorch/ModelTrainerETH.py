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
        # [NeMO] This call "transforms" the model into a quantization-aware one, which is printed immediately afterwards.
        self.model = nemo.transform.quantize_pact(self.model)
        logging.info("[ModelTrainer] Model: %s", self.model)
        # [NeMO] NeMO re-training usually converges better using an Adam optimizer, and a smaller learning rate
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.regime['lr']),
                                     weight_decay=float(self.regime['weight_decay']))

        # [NeMO] DNNs that do not employ batch normalization layers nor have clipped activations (e.g. ReLU6) require
        # an initial calibration to transfer to a quantization-aware version. This is used to calibrate the scaling
        # parameters of quantization-aware activation layers to the point defined by the maximum activation value
        # seen during a validation run. DNNs that employ BN or ReLU6 (or both) do not require this operation, as their
        # activations are already statistically bounded in terms of dynamic range.
        logging.info("[ModelTrainer] Gather statistics for non batch-normed activations")
        self.model.set_statistics_act()
        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            validation_loader)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        self.model.unset_statistics_act()
        self.model.reset_alpha_act()
        logging.info("[ModelTrainer] %.2f" % acc)

        precision_rule = self.regime['relaxation']

        # [NeMO] The evaluation engine performs a simple grid search to decide, among the possible quantization configurations,
        # which one is the most promising step for the relaxation procedure. It uses an internal heuristic binning validation
        # results in top-bin (high accuracy), middle-bin (reduced accuracy, but not garbage) and bottom-bin (garbage results).
        # It typically selects a step from the middle-bin to maximize training speed without sacrificing the final results.
        evale = nemo.evaluation.EvaluationEngine(self.model, precision_rule=precision_rule,
                                                 validate_fn=self.ValidateSingleEpoch,
                                                 validate_data=validation_loader)
        # while evale.step():
        #     valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
        #         validation_loader)
        #     acc = torch.tensor(float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi))
        #     evale.report(acc)
        #     logging.info("[ModelTrainer] %.1f-bit W, %.1f-bit x: %.2f" % (
        #         evale.wgrid[evale.idx], evale.xgrid[evale.idx], acc))
        #Wbits, xbits = evale.get_next_config(upper_threshold=0.97)
        Wbits = 16
        xbits = 16
        precision_rule['0']['W_bits'] = min(Wbits, precision_rule['0']['W_bits'])
        precision_rule['0']['x_bits'] = min(xbits, precision_rule['0']['x_bits'])
        logging.info("[ModelTrainer] Choosing %.1f-bit W, %.1f-bit x for first step" % (
            precision_rule['0']['W_bits'], precision_rule['0']['x_bits']))

        # [NeMO] The relaxation engine can be stepped to automatically change the DNN precisions and end training if the final
        # target has been achieved.
        self.relax = nemo.relaxation.RelaxationEngine(self.model, optimizer, criterion=None, trainloader=None,
                                                 precision_rule=precision_rule, reset_alpha_weights=False,
                                                 min_prec_dict=None, evaluator=evale)



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
            if self.args.quantize:
                change_prec, ended = self.relax.step(loss_epoch_m1, epoch, None)
            if ended:
                break

            train_loss_x, train_loss_y, train_loss_z, train_loss_phi = self.TrainSingleEpoch(training_generator)

            valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
                validation_generator)

            valid_loss = valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi
            scheduler.step(valid_loss)

            gt_labels = torch.tensor(gt_labels, dtype=torch.float32)
            y_pred = torch.tensor(y_pred, dtype=torch.float32)
            MSE, MAE, r_score = metrics.Update(y_pred, gt_labels,
                                               [train_loss_x, train_loss_y, train_loss_z, train_loss_phi],
                                               [valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi])

            logging.info('[ModelTrainer] Validation MSE: {}'.format(MSE))
            logging.info('[ModelTrainer] Validation MAE: {}'.format(MAE))
            logging.info('[ModelTrainer] Validation r_score: {}'.format(r_score))

            checkpoint_filename = self.folderPath + 'Dronet-{:03d}.pt'.format(epoch)
            early_stopping(valid_loss, self.model, epoch, checkpoint_filename)
            if early_stopping.early_stop:
                logging.info("[ModelTrainer] Early stopping")
                break

        MSEs = metrics.GetMSE()
        MAEs = metrics.GetMAE()
        r_score = metrics.Getr2_score()
        y_pred_viz = metrics.GetPred()
        gt_labels_viz = metrics.GetLabels()
        train_losses_x, train_losses_y, train_losses_z, train_losses_phi, valid_losses_x, valid_losses_y, valid_losses_z, valid_losses_phi = metrics.GetLosses()

        DataVisualization.desc = "Train_"
        DataVisualization.PlotLoss(train_losses_x, train_losses_y, train_losses_z, train_losses_phi , valid_losses_x, valid_losses_y, valid_losses_z, valid_losses_phi)
        DataVisualization.PlotMSE(MSEs)
        DataVisualization.PlotMAE(MAEs)
        DataVisualization.PlotR2Score(r_score)

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
        outputs = torch.t(outputs)
        outputs = outputs.cpu().numpy()
        return outputs

    def Predict(self, test_generator):

        metrics = Metrics()

        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            test_generator)

        gt_labels = torch.tensor(gt_labels, dtype=torch.float32)
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
        MSE, MAE, r_score = metrics.Update(y_pred, gt_labels,
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
        logging.info('[ModelTrainer] Test r_score: {}'.format(r_score))

    def Infer(self, live_generator):

        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            live_generator)

        return y_pred

