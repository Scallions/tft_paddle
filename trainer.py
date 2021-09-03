# -*- coding: utf-8 -*-
# ---------------------
import os
from time import time
import numpy as np
import paddle
from paddle import optimizer
from paddle.io import DataLoader
from tft_model_paddle import TFT
from conf import Conf
from dataset.ts_dataset import TSDataset
from progress_bar import ProgressBar
from utils import QuantileLoss, symmetric_mean_absolute_percentage_error, unnormalize_tensor, plot_temporal_serie
from data_formatters.utils import logger_config
from scheduler import CosineAnnealingDecay
import data_formatters.utils as utils

if not os.path.isdir('experiment'):
    os.makedirs('experiment')
logger = logger_config(log_path='experiment/TFTransformer_log9.txt', logging_name='TFTransformer')

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

        self.loss = QuantileLoss(cnf.quantiles)

        # init train loader
        self.train_loader = DataLoader(
            dataset=dataset_train, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=True)

        # init test loader
        self.test_loader = DataLoader(
            dataset=dataset_test, batch_size=cnf.batch_size,
            num_workers=cnf.n_workers, shuffle=False)
        self.log_freq = len(self.train_loader)
        # init optimizer
        length = len(self.train_loader)
        T_period = [length for _ in range(self.cnf.all_params['num_epochs'])]
        restarts = [length * i for i in range(1, self.cnf.all_params['num_epochs']+1)]
        weights = [1 for _ in range(self.cnf.all_params['num_epochs'])]
        # init optimizer
        #self.scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=self.cnf.all_params['lr'], T_max=len(self.train_loader)*2, verbose=False)
        self.scheduler = CosineAnnealingDecay(learning_rate=self.cnf.all_params['lr'],
                                              T_period=T_period,
                                              restarts=restarts,
                                              weights=weights,
                                              eta_min=0.00001)
        self.optimizer = paddle.optimizer.Adam(learning_rate=self.scheduler,
                                               parameters=self.model.parameters(),
                                               grad_clip=paddle.nn.ClipGradByNorm(self.cnf.all_params['max_gradient_norm']))

        # init logging stuffs
        self.log_path = cnf.exp_log_path

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
            # self.optimizer.clear_grad()
            # Feed input to the model
            x = sample['inputs'].astype('float32')
            output = self.model.forward(x)
            # Compute Loss
            loss, _ = self.loss(output.squeeze(), sample['outputs'].squeeze().astype('float32'))
            loss.backward()
            self.train_losses.append(loss.item())
            self.optimizer.step()
            self.optimizer.clear_grad()
            self.scheduler.step()
            cur_lr = self.scheduler.get_lr()
            if step % self.cnf.all_params['log_step'] == 0:
                logger.info('[TRAIN] Epoch {} \t Iter {} \t Lr {} \t Loss {:.6f}'.format(self.epoch, step+1, cur_lr, loss.item()))

        # log average loss of this epoch
        mean_epoch_loss = np.mean(self.train_losses)
        self.train_losses = []


    def test(self):
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

            # Compute loss
            loss, _ = self.loss(y_pred, y)
            # smape = symmetric_mean_absolute_percentage_error(output[:, :, 1],
            #                                                  sample['outputs'][:, :, 0])

            # De-Normalize to compute metrics

            target = unnormalize_tensor(self.data_formatter, y, sample['identifier'][0][0])
            #p10_forecast = unnormalize_tensor(self.data_formatter, y_pred[:, :, 0], sample['identifier'][0][0])
            #p50_forecast = unnormalize_tensor(self.data_formatter, y_pred[:, :, 1], sample['identifier'][0][0])
            p90_forecast = unnormalize_tensor(self.data_formatter, y_pred[:, :, 2], sample['identifier'][0][0])

            # Compute metrics
            #self.test_losses['p10'].append(self.loss.numpy_normalised_quantile_loss(p10_forecast, target, 0.1))
            #self.test_losses['p50'].append(self.loss.numpy_normalised_quantile_loss(p50_forecast, target, 0.5))
            self.test_losses['p90'].append(self.loss.numpy_normalised_quantile_loss(p90_forecast, target, 0.9))

            self.test_loss.append(loss.item())
            #self.test_smape.append(smape)
 
        # Log stuff
        # for k in self.test_losses.keys():
        mean_test_loss = np.mean(self.test_losses['p90'])
        logger.info(f'[TEST] AVG p90 Loss on TEST-set: {mean_test_loss:.6f} │ T: {time() - t:.2f} s')

        # log log log
        mean_test_loss = np.mean(self.test_loss)
        #mean_smape = np.mean(self.test_smape)
        logger.info(f'[TEST] AVG Loss on TEST-set: {mean_test_loss:.6f} │ T: {time() - t:.2f} s')
        #print(f'\t● AVG SMAPE on TEST-set: {mean_smape:.6f} │ T: {time() - t:.2f} s')


        # save best model
        if self.best_test_loss is None or mean_test_loss < self.best_test_loss:
            self.best_test_loss = mean_test_loss
            paddle.save(self.model.state_dict(), self.log_path / self.cnf.exp_name + '_best.pdparams')
            paddle.save(self.optimizer.state_dict(),self.log_path / self.cnf.exp_name + '_best.pdopt')

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
