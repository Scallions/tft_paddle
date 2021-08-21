# -*- coding: utf-8 -*-
# ---------------------
import os
from time import time
import numpy as np
import paddle
from paddle import optimizer
from paddle.io import DataLoader
from conf import Conf
from dataset.ts_dataset import TSDataset
from tft_model import TFT
from progress_bar import ProgressBar
from utils import QuantileLoss, symmetric_mean_absolute_percentage_error, unnormalize_tensor, plot_temporal_serie
import data_formatters.utils as utils



class TS(object):
    """
    Class for loading and test the pre-trained model
    """

    def __init__(self, cnf):
        # type: (Conf) -> Trainer

        self.cnf = cnf
        self.data_formatter = utils.make_data_formatter(cnf.ds_name)

        loader = TSDataset
        dataset_test = loader(self.cnf, self.data_formatter)
        dataset_test.test()

        # init model
        model_choice = self.cnf.all_params["model"]
        if model_choice == "transformer":
            # Baseline transformer
            self.model = TFT(self.cnf.all_params)
        else:
            raise NameError


        # init optimizer
        self.optimizer = paddle.optimizer.Adam(learning_rate=self.cnf.all_params['lr'],
                                               parameters=self.model.parameters(),
                                               grad_clip=self.cnf.all_params['max_gradient_norm'])
        self.loss = QuantileLoss(cnf.quantiles)

        # init test loader
        self.test_loader = DataLoader(
            dataset=dataset_test, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=False)

        # init logging stuffs
        self.log_path = cnf.exp_log_path
        self.log_freq = len(self.test_loader)
        self.train_losses = []
        self.test_loss = []
        self.test_losses = {'p10': [], 'p50': [], 'p90': []}
        self.test_smape = []

        # starting values
        self.epoch = 0
        self.best_test_loss = None

        # init progress bar
        self.progress_bar = ProgressBar(max_step=self.log_freq, max_epoch=self.cnf.epochs)

        # possibly load checkpoint
        self.resume(self.model, self.optimizer, self.log_path)

        print("Finished preparing datasets.")

    def resume(self, model, optimizer, resume_model):
        if resume_model is not None:
            print('Resume model from {}'.format(resume_model))
            if os.path.exists(resume_model):
                resume_model = os.path.normpath(resume_model)
                ckpt_path = os.path.join(resume_model, 'model.pdparams')
                para_state_dict = paddle.load(ckpt_path)
                ckpt_path = os.path.join(resume_model, 'model.pdopt')
                opti_state_dict = paddle.load(ckpt_path)
                model.set_state_dict(para_state_dict)
                optimizer.set_state_dict(opti_state_dict)

                iter = resume_model.split('_')[-1]
                iter = int(iter)
                return iter
            else:
                raise ValueError(
                    'Directory of the model needed to resume is not Found: {}'.
                        format(resume_model))
        else:
            print('No model needed to resume.')

    def test(self):
        """
        Quick test and plot prediction without saving or logging stuff on tensorboarc
        """
        with paddle.no_grad():
            self.model.eval()
            p10_forecast, p10_forecast, p90_forecast, target = None, None, None, None

            t = time()
            for step, sample in enumerate(self.test_loader):

                # Hide future predictions from input vector, set to 0 (or 1) values where timestep > encoder_steps
                steps = self.cnf.all_params['num_encoder_steps']
                pred_len = sample['outputs'].shape[1]
                x = sample['inputs'].float()
                x[:, steps:, 0] = 1

                # Feed input to the model
                if self.cnf.all_params["model"] == "transformer" or self.cnf.all_params["model"] == "grn_transformer":

                    # Auto-regressive prediction
                    for i in range(pred_len):
                        output = self.model.forward(x)
                        x[:, steps + i, 0] = output[:, i, 1]
                    output = self.model.forward(x)

                elif self.cnf.all_params["model"] == "tf_transformer":
                    output, _, _ = self.model.forward(x)
                else:
                    raise NameError

                output = output.squeeze()
                y, y_pred = sample['outputs'].squeeze().astype('float32'), output

                # Compute loss
                loss, _ = self.loss(y_pred, y)
                smape = symmetric_mean_absolute_percentage_error(output[:, :, 1],
                                                                 sample['outputs'][:, :, 0])

                # De-Normalize to compute metrics
                target = unnormalize_tensor(self.data_formatter, y, sample['identifier'][0][0])
                p10_forecast = unnormalize_tensor(self.data_formatter, y_pred[..., 0], sample['identifier'][0][0])
                p50_forecast = unnormalize_tensor(self.data_formatter, y_pred[..., 1], sample['identifier'][0][0])
                p90_forecast = unnormalize_tensor(self.data_formatter, y_pred[..., 2], sample['identifier'][0][0])

                # Compute metrics
                self.test_losses['p10'].append(self.loss.numpy_normalised_quantile_loss(p10_forecast, target, 0.1))
                self.test_losses['p50'].append(self.loss.numpy_normalised_quantile_loss(p50_forecast, target, 0.5))
                self.test_losses['p90'].append(self.loss.numpy_normalised_quantile_loss(p90_forecast, target, 0.9))

                self.test_loss.append(loss.item())
                self.test_smape.append(smape)

            # Plot serie prediction
            p1, p2, p3, target = np.expand_dims(p10_forecast, axis=-1), np.expand_dims(p50_forecast, axis=-1), \
                                 np.expand_dims(p90_forecast, axis=-1), np.expand_dims(target, axis=-1)
            p = np.concatenate((p1, p2, p3), axis=-1)
            plot_temporal_serie(p, target)

            # Log stuff
            for k in self.test_losses.keys():
                mean_test_loss = np.mean(self.test_losses[k])
                print(f'\t● AVG {k} Loss on TEST-set: {mean_test_loss:.6f} │ T: {time() - t:.2f} s')

            # log log log
            mean_test_loss = np.mean(self.test_loss)
            mean_smape = np.mean(self.test_smape)
            print(f'\t● AVG Loss on TEST-set: {mean_test_loss:.6f} │ T: {time() - t:.2f} s')
            print(f'\t● AVG SMAPE on TEST-set: {mean_smape:.6f} │ T: {time() - t:.2f} s')
