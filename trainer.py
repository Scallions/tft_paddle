# -*- coding: utf-8 -*-
# ---------------------
import os
from time import time
import numpy as np
import paddle
from paddle import optimizer
from paddle.io import DataLoader
from model.tft_model import TFT
from config import Conf
from data.ts_dataset import TSDataset
from utils.progress_bar import ProgressBar
from utils.utils import QuantileLoss, symmetric_mean_absolute_percentage_error, unnormalize_tensor, plot_temporal_serie
from data.utils import logger_config
import data.utils as utils


logger = logger_config(log_path='experiment/TFTransformer_log.txt', logging_name='TFTransformer')

class Trainer(object):
    """
    Class for training and test the model
    """

    def __init__(self, cnf):


        #paddle.set_num_threads(3)

        self.cnf = cnf
        self.data_formatter = utils.make_data_formatter(cnf.ds_name)

        loader = TSDataset

        # init dataset
        dataset_train = loader(self.cnf, self.data_formatter)
        dataset_train.train()
        dataset_test = loader(self.cnf, self.data_formatter)
        dataset_test.test()

        # init model
        model_choice = self.cnf.all_params["model"]
        print('model_choice:',model_choice)

        if model_choice == "tf_transformer":
            # Temporal fusion transformer
            self.model = TFT(self.cnf.all_params)
        else:
            raise NameError

        # init optimizer
        # self.optimizer = paddle.optimizer.Adam(learning_rate=self.cnf.all_params['lr'],
        #                                        parameters=self.model.parameters(),
        #                                        grad_clip=paddle.nn.ClipGradByNorm(self.cnf.all_params['max_gradient_norm']))
        
        self.loss = QuantileLoss(cnf.quantiles)

        # init train loader
        self.train_loader = DataLoader(
            dataset=dataset_train, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=True)

        # init test loader
        self.test_loader = DataLoader(
            dataset=dataset_test, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=False)

        # init logging stuffs
        self.log_path = cnf.exp_log_path

        self.scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=self.cnf.all_params['lr'], T_max=len(self.train_loader), eta_min=0.00001, verbose=False)
        self.optimizer = paddle.optimizer.Adam(learning_rate=self.scheduler,
                                               parameters=self.model.parameters(),
                                               grad_clip=paddle.nn.ClipGradByNorm(self.cnf.all_params['max_gradient_norm']))
        self.log_freq = len(self.train_loader)
        self.train_losses = []
        self.test_loss = []
        self.test_losses = {'p90': []}
        self.test_smape = []

        # starting values
        self.epoch = 0
        self.best_test_loss = None

        # init progress bar
        self.progress_bar = ProgressBar(max_step=self.log_freq, max_epoch=self.cnf.epochs)

        # possibly load checkpoint
        # self.resume(self.model, self.optimizer, self.log_path)

        print("Finished preparing datasets.")

    def save_model(self, model, optimizer, save_dir, epoch):
        current_save_dir = os.path.join(save_dir, "epoch_{}".format(epoch))
        nranks = paddle.distributed.ParallelEnv().nranks
        if not os.path.isdir(current_save_dir):
            os.makedirs(current_save_dir)
        if nranks > 1:
            paddle.distributed.fleet.init(is_collective=True)
            optimizer = paddle.distributed.fleet.distributed_optimizer(
                optimizer)  # The return is Fleet object
            model = paddle.distributed.fleet.distributed_model(model)
        paddle.save(model.state_dict(),
                    os.path.join(current_save_dir, 'model.pdparams'))
        paddle.save(optimizer.state_dict(),
                    os.path.join(current_save_dir, 'model.pdopt'))

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

    def train(self):
        """
        train model for one epoch on the Training-Set.
        """
        start_time = time()
        self.model.train()

        #times = []
        for step, sample in enumerate(self.train_loader):
            t = time()
            self.optimizer.clear_grad()
            # Feed input to the model
            x = sample['inputs'].astype('float32')
            output = self.model.forward(x)
            # Compute Loss
            loss, _ = self.loss(output.squeeze(), sample['outputs'].squeeze().astype('float32'))
            if loss.isnan().any().item() == True:
                exit()
            loss.backward()
            self.train_losses.append(loss.item())
            self.optimizer.step()

            if step % self.cnf.all_params['log_step'] == 0:
                logger.info('[TRAIN] Epoch {} \t Iter {} \t Loss {:.6f}'.format(self.epoch, step+1, loss.item()))
                # self.test()
            # print an incredible progress bar
            # times.append(time() - t)
            # if self.cnf.log_each_step or (not self.cnf.log_each_step and self.progress_bar.progress == 1):
            #     print(f'\r{self.progress_bar} '
            #           f'©¦ Loss: {np.mean(self.train_losses):.6f} '
            #           f'©¦ ?: {1 / np.mean(times):5.2f} step/s', end='')
            # self.progress_bar.inc()

        # log average loss of this epoch
        mean_epoch_loss = np.mean(self.train_losses)
        self.train_losses = []

        # log epoch duration
        # print(f' ©¦ T: {time() - start_time:.2f} s')


    def test(self, final=False):
        """
        test model on the Test-Set
        """
        self.model.eval()
        output, sample = None, None

        t = time()
        for step, sample in enumerate(self.test_loader):

            # Hide future predictions from input vector, set to 0 (or 1) values where timestep > encoder_steps
            steps = self.cnf.all_params['num_encoder_steps']
            pred_len = sample['outputs'].shape[1]
            x = sample['inputs'].astype('float32')
            x[:, steps:, 0] = 1

            # Feed input to the model
            if self.cnf.all_params["model"] == "tf_transformer":
                output = self.model.forward(x)
            else:
                raise NameError

            output = output.squeeze()
            y, y_pred = sample['outputs'].squeeze().astype('float32'), output
            if y_pred.isnan().any().item() == True:
                exit()

            # Compute loss
            loss, _ = self.loss(y_pred, y)
            # smape = symmetric_mean_absolute_percentage_error(output[:, :, 1],
            #                                                  sample['outputs'][:, :, 0])

            # De-Normalize to compute metrics

            target = unnormalize_tensor(self.data_formatter, y, sample['identifier'][0][0])
            # p10_forecast = unnormalize_tensor(self.data_formatter, y_pred[:, :, 0], sample['identifier'][0][0])
            # p50_forecast = unnormalize_tensor(self.data_formatter, y_pred[:, :, 1], sample['identifier'][0][0])
            p90_forecast = unnormalize_tensor(self.data_formatter, y_pred[:, :, 2], sample['identifier'][0][0])

            # Compute metrics
            # self.test_losses['p10'].append(self.loss.numpy_normalised_quantile_loss(p10_forecast, target, 0.1))
            # self.test_losses['p50'].append(self.loss.numpy_normalised_quantile_loss(p50_forecast, target, 0.5))
            self.test_losses['p90'].append(self.loss.numpy_normalised_quantile_loss(p90_forecast, target, 0.9))

            self.test_loss.append(loss.item())
            #self.test_smape.append(smape)
            #if step % 200 == 0:
            #    logger.info('[EVAL] Epoch {} \t Iter {} \t mean P90 Loss {:.6f}'.format(self.epoch, step+1, np.mean(self.test_losses['p90'])))
        # Log stuff
        for k in self.test_losses.keys():
            if final:
                mean_test_loss = np.mean(self.test_losses[k])
                """@nni.report_final_result(mean_test_loss)"""
            else:
                mean_test_loss = np.mean(self.test_losses[k])
                """@nni.report_intermediate_result(mean_test_loss)"""
            self.test_losses[k] = []
            logger.info(f'\t¡ñ AVG {k} Loss on TEST-set: {mean_test_loss:.6f} ©¦ T: {time() - t:.2f} s')

        # log log log
        mean_test_loss = np.mean(self.test_loss)
        #mean_smape = np.mean(self.test_smape)
        logger.info(f'\t¡ñ AVG Loss on TEST-set: {mean_test_loss:.6f} ©¦ T: {time() - t:.2f} s')
        #print(f'\t¡ñ AVG SMAPE on TEST-set: {mean_smape:.6f} ©¦ T: {time() - t:.2f} s')


        # save best model
        if self.best_test_loss is None or mean_test_loss < self.best_test_loss:
            self.best_test_loss = mean_test_loss
            paddle.save(self.model.state_dict(), self.log_path / self.cnf.exp_name + '_best.pdparams')
        self.model.train()

    def run(self):
        """
        start model training procedure (train > test > checkpoint > repeat)
        """
        for _ in range(self.epoch, self.cnf.epochs):
            self.train()

            with paddle.no_grad():
                self.test()

            self.epoch += 1
            self.save_model(self.model, self.optimizer, self.log_path, self.epoch)

        
        # final
        with paddle.no_grad():
            self.test(final=True)
