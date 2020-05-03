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
import CSVUtils as utils

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
            param = dict(self.model.named_parameters())
            fp_params = list({k: v for k, v in param.items() if k[:3] == "fc."}.values())
            qnt_params = list({k: v for k, v in param.items() if k[:3] != "fc."}.values())
            self.optimizer = torch.optim.Adam((
                {'params': qnt_params, 'lr': float(regime['lr']), 'weight_decay': float(regime['weight_decay'])},
                {'params': fp_params, 'lr': float(regime['lr']), 'weight_decay': float(regime['weight_decay'])}
            ))
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.relax = None
        self.folderPath = "Models/"

    def GetModel(self):
        return self.model

    # Francesco's code from https://github.com/FrancescoConti/FrontNetPorting/
    def Deploy(self, validation_loader, h, w, prec_dict=None):

        logging.info("[ModelTrainer] Model: %s", self.model)
        self.model.change_precision(bits=1, reset_alpha=False, min_prec_dict=prec_dict)

        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            validation_loader)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        logging.info("[ModelTrainer]: FakeQuantized network: %f" % acc)

        # qd_stage requires NEMO>=0.0.3
        # input is in [0,255], so eps_in=1 (smallest representable amount in the input) and there is no input bias
        self.model.qd_stage(eps_in=1.0)
        bin_qd, bout_qd, (valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred,
                          gt_labels) = nemo.utils.get_intermediate_activations(self.model, self.ValidateSingleEpoch,
                                                                               validation_loader)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        logging.info("[ModelTrainer]: QuantizedDeployable network: %f" % acc)

        # id_stage requires NEMO>=0.0.3
        self.model.id_stage()
        bin_id, bout_id, (valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred,
                          gt_labels) = nemo.utils.get_intermediate_activations(self.model, self.ValidateSingleEpoch,
                                                                               validation_loader, integer=True)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        logging.info("[ModelTrainer]: IntegerDeployable: %f" % acc)

        # export model
        try:
            os.makedirs(self.model.name)
        except Exception:
            pass
        nemo.utils.export_onnx(self.model.name + "/model_int.onnx", self.model, self.model, (1, h, w), perm=None)

        # export golden outputs
        b_in = bin_id
        b_out = bout_id
        try:
            os.makedirs(self.model.name + '/golden')
        except Exception:
            pass
        from collections import OrderedDict
        dory_dict = OrderedDict([])
        bidx = 0
        for n, m in self.model.named_modules():
            try:

                actbuf = b_in[n][0][bidx].permute((1, 2, 0))
            except RuntimeError:
                actbuf = b_in[n][0][bidx]
            np.savetxt(self.model.name + "/golden/golden_input_%s.txt" % n, actbuf.cpu().numpy().flatten(),
                       header="input (shape %s)" % (list(actbuf.shape)), fmt="%.3f", delimiter=',', newline=',\n')
        for n, m in self.model.named_modules():
            try:
                actbuf = b_out[n][bidx].permute((1, 2, 0))
            except RuntimeError:
                actbuf = b_out[n][bidx]
            np.savetxt(self.model.name + "/golden/golden_%s.txt" % n, actbuf.cpu().numpy().flatten(),
                       header="%s (shape %s)" % (n, list(actbuf.shape)), fmt="%.3f", delimiter=',', newline=',\n')


#Francesco's code from https://github.com/FrancescoConti/FrontNetPorting/

    def TrainQuantized(self, train_loader, validation_loader, h, w, epochs=100, relaxation=False):

        print(self.model.name)

        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            validation_loader)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        logging.info("[ModelTrainer]: Before quantization process: %f" % acc)

        # [NeMO] This call "transforms" the model into a quantization-aware one, which is printed immediately afterwards.
        self.model = nemo.transform.quantize_pact(self.model,
                                                  dummy_input=torch.ones((1, 1, h, w)).to(self.device))  # .cuda()
        logging.info("[ModelTrainer] Model: %s", self.model)
        self.model.change_precision(bits=20)

        # calibration
        self.model.reset_alpha_weights()
        self.model.set_statistics_act()
        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            validation_loader)
        _ = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        self.model.unset_statistics_act()
        self.model.reset_alpha_act()
        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            validation_loader)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        logging.info("[ModelTrainer]: After calibration process: %f" % acc)

        # [NeMO] NeMO re-training usually converges better using an Adam optimizer, and a smaller learning rate
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.regime['lr']),
                                     weight_decay=float(self.regime['weight_decay']))

        precision_rule = self.regime['relaxation']

        # [NeMO] The relaxation engine can be stepped to automatically change the DNN precisions and end training if the final
        # target has been achieved.
        if relaxation:
            relax = nemo.relaxation.RelaxationEngine(self.model, optimizer, criterion=None, trainloader=None,
                                                     precision_rule=precision_rule, reset_alpha_weights=False,
                                                     min_prec_dict=None, evaluator=None)

        # directly try to go to 7x8b
        # prec_dict = {
        #     'fc': {'W_bits': 15}
        # }
        prec_dict = {}
        if relaxation:
            self.model.change_precision(bits=12, min_prec_dict=prec_dict)
            self.model.change_precision(bits=12, scale_activations=False, min_prec_dict=prec_dict)
        else:
            self.model.change_precision(bits=7, min_prec_dict=prec_dict)
            self.model.change_precision(bits=7, scale_activations=False, min_prec_dict=prec_dict)
        self.ipython = False

        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            validation_loader)
        acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)
        logging.info("[ModelTrainer]: Before fine-tuning: %f" % acc)

        loss_epoch_m1 = 1e3
        #loss_epoch_m1 = acc

        train_qnt = False

        best = 0.

        # self.model.enable_prefc = True

        for epoch in range(1, epochs):


            train_qnt = not train_qnt
            print(self.model.name)

            change_prec = False
            ended = False
            if relaxation:
                change_prec, ended = relax.step(loss_epoch_m1, epoch, checkpoint_name=self.model.name)
                # If I try to run with Relaxation = True, I get exception here. This is because loss_epoch_m1 > self.precision_rule['divergence_abs_threshold']
                # and the code tries to load a checkpoint that does not exist.....
                # Setting loss_epoch_m1 = acc solves it, but who knows if it's correct
            else:
                self.optimizer.param_groups[0]['lr'] *= float(self.regime['lr_decay'])
                self.optimizer.param_groups[1]['lr'] *= float(self.regime['lr_decay'])
            if ended:
                break


            valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
                validation_loader)
            acc = float(1) / (valid_loss_x + valid_loss_y + valid_loss_z + valid_loss_phi)

            logging.info(
                "[ModelTrainer] Epoch: %d Train loss: %.2f Accuracy: %.2f%%" % (epoch, loss_epoch_m1, acc * 100.))

            if acc > best:
                nemo.utils.save_checkpoint(self.model, self.optimizer, epoch, acc, checkpoint_name=self.model.name,
                                           checkpoint_suffix='best')
                best = acc

            if self.ipython:
                import IPython;
                IPython.embed()
        nemo.utils.save_checkpoint(self.model, self.optimizer, epoch, acc, checkpoint_name=self.model.name,
                                   checkpoint_suffix='final')

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
            #loss = self.criterion(outputs, batch_targets)


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

    # Francesco's code from https://github.com/FrancescoConti/FrontNetPorting/.

    def ValidateSingleEpoch(self, validation_generator, integer=False):

        self.model.eval()

        valid_loss = {
            'tot': RunningAverage(),
            'x': RunningAverage(),
            'y': RunningAverage(),
            'z': RunningAverage(),
            'phi': RunningAverage()
        }
        mse_loss = {
            'tot': RunningAverage(),
            'x': RunningAverage(),
            'y': RunningAverage(),
            'z': RunningAverage(),
            'phi': RunningAverage()
        }
        mae_loss = {
            'tot': RunningAverage(),
            'x': RunningAverage(),
            'y': RunningAverage(),
            'z': RunningAverage(),
            'phi': RunningAverage()
        }

        y_pred = []
        gt_labels = []
        with torch.no_grad():
            for batch_samples, batch_targets in validation_generator:
                gt_labels.extend(batch_targets.cpu().numpy())
                batch_targets = batch_targets.to(self.device)
                batch_samples = batch_samples.to(self.device)
                outputs = self.model(batch_samples)
                if integer:
                    eps_fcin = (self.model.layer3.relu2.alpha / (2 ** self.model.layer3.relu2.precision.get_bits() - 1))
                    eps_fcout = self.model.fc.get_output_eps(eps_fcin)
                    # workaround because PACT_Linear is not properly quantizing biases!
                    outputs = [(o - self.model.fc.bias[i]) * eps_fcout + self.model.fc.bias[i] for i, o in
                               enumerate(outputs)]

                loss_x = self.criterion(outputs[0], (batch_targets[:, 0]).view(-1, 1))
                loss_y = self.criterion(outputs[1], (batch_targets[:, 1]).view(-1, 1))
                loss_z = self.criterion(outputs[2], (batch_targets[:, 2]).view(-1, 1))
                loss_phi = self.criterion(outputs[3], (batch_targets[:, 3]).view(-1, 1))
                loss = loss_x + loss_y + loss_z + loss_phi  # does it make sense for SmoothL1Loss?
                valid_loss['tot'].update(loss)
                valid_loss['x'].update(loss_x)
                valid_loss['y'].update(loss_y)
                valid_loss['z'].update(loss_z)
                valid_loss['phi'].update(loss_phi)

                loss_x = nn.functional.l1_loss(outputs[0], (batch_targets[:, 0]).view(-1, 1))
                loss_y = nn.functional.l1_loss(outputs[1], (batch_targets[:, 1]).view(-1, 1))
                loss_z = nn.functional.l1_loss(outputs[2], (batch_targets[:, 2]).view(-1, 1))
                loss_phi = nn.functional.l1_loss(outputs[3], (batch_targets[:, 3]).view(-1, 1))
                loss = loss_x + loss_y + loss_z + loss_phi
                mae_loss['tot'].update(loss)
                mae_loss['x'].update(loss_x)
                mae_loss['y'].update(loss_y)
                mae_loss['z'].update(loss_z)
                mae_loss['phi'].update(loss_phi)

                loss_x = nn.functional.mse_loss(outputs[0], (batch_targets[:, 0]).view(-1, 1))
                loss_y = nn.functional.mse_loss(outputs[1], (batch_targets[:, 1]).view(-1, 1))
                loss_z = nn.functional.mse_loss(outputs[2], (batch_targets[:, 2]).view(-1, 1))
                loss_phi = nn.functional.mse_loss(outputs[3], (batch_targets[:, 3]).view(-1, 1))
                loss = loss_x + loss_y + loss_z + loss_phi
                mse_loss['tot'].update(loss)
                mse_loss['x'].update(loss_x)
                mse_loss['y'].update(loss_y)
                mse_loss['z'].update(loss_z)
                mse_loss['phi'].update(loss_phi)

                outputs = torch.stack(outputs, 0)
                outputs = torch.squeeze(outputs)
                outputs = torch.t(outputs)
                y_pred.extend(outputs.cpu().numpy())

            logging.info("[ModelTrainer] Mean validation loss {:2f}: x={:2f}, y={:2f}, z={:2f}, phi={:2f}".format(
                valid_loss['tot'].value, valid_loss['x'].value, valid_loss['y'].value,
                valid_loss['z'].value,
                valid_loss['phi'].value))
            logging.info("[ModelTrainer] MSE {:2f}: x={:2f}, y={:2f}, z={:2f}, phi={:2f}".format(mse_loss['tot'].value,
                                                                                                 mse_loss['x'].value,
                                                                                                 mse_loss['y'].value,
                                                                                                 mse_loss['z'].value,
                                                                                                 mse_loss['phi'].value))
            logging.info("[ModelTrainer] MAE {:2f}: x={:2f}, y={:2f}, z={:2f}, phi={:2f}".format(mae_loss['tot'].value,
                                                                                                 mae_loss['x'].value,
                                                                                                 mae_loss['y'].value,
                                                                                                 mae_loss['z'].value,
                                                                                                 mae_loss['phi'].value))

        return valid_loss['x'].value, valid_loss['y'].value, valid_loss['z'].value, valid_loss[
            'phi'].value, y_pred, gt_labels


    def Train(self, training_generator, validation_generator):

        metrics = Metrics()
        early_stopping = EarlyStopping(patience=10, verbose=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=np.sqrt(0.1),
                                                                    patience=5, verbose=False,
                                                                    threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                                    min_lr=0.1e-6, eps=1e-08)

        for epoch in range(1, self.args.epochs + 1):
            logging.info("[ModelTrainer] Starting Epoch {}".format(epoch))

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

        utils.SaveModelResultsToCSV(MSEs, MAEs, r2_score, gt_labels_viz, y_pred_viz, "Results/train")

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

        logging.info('[ModelTrainer] Test MSE: [{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}]'.format(MSE[0], MSE[1], MSE[2], MSE[3]))
        logging.info('[ModelTrainer] Test MAE: [{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}]'.format(MAE[0], MAE[1], MAE[2], MAE[3]))
        logging.info('[ModelTrainer] Test r2_score: [{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}]'.format(r2_score[0], r2_score[1], r2_score[2],
                                                                                  r2_score[3]))

    def Test(self, test_generator):

        metrics = Metrics()

        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            test_generator)

        outputs = y_pred
        outputs = np.reshape(outputs, (-1, 4))
        labels = gt_labels
        y_pred = np.reshape(y_pred, (-1, 4))
        gt_labels = torch.tensor(gt_labels, dtype=torch.float32)
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
        MSE, MAE, r2_score = metrics.Update(y_pred, gt_labels,
                                           [0, 0, 0, 0],
                                           [valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi])

        logging.info('[ModelTrainer] Test MSE: [{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}]'.format(MSE[0], MSE[1], MSE[2], MSE[3]))
        logging.info('[ModelTrainer] Test MAE: [{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}]'.format(MAE[0], MAE[1], MAE[2], MAE[3]))
        logging.info('[ModelTrainer] Test r2_score: [{0:.4f}, {1:.4f}, {2:.4f}, {3:.4f}]'.format(r2_score[0], r2_score[1], r2_score[2], r2_score[3] ))


        return MSE, MAE, r2_score, outputs, labels


    def Infer(self, live_generator):

        valid_loss_x, valid_loss_y, valid_loss_z, valid_loss_phi, y_pred, gt_labels = self.ValidateSingleEpoch(
            live_generator)

        return y_pred

