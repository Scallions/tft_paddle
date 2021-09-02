# -*- coding: utf-8 -*-
# ---------------------

import click
from paddle.batch import batch
from config import Conf
from trainer import Trainer
from inference import TS



@click.command()
@click.option('--exp_name', type=str, default='electricity')
@click.option('--conf_file_path', type=str, default='config/electricity.yaml')
@click.option('--seed', type=int, default=None)
@click.option('--inference', type=bool, default=False)
def main(exp_name, conf_file_path, seed, inference):

    # if `exp_name` is None,
    # ask the user to enter it
    if exp_name is None:
        exp_name = click.prompt('▶ experiment name', default='default')

    # if `exp_name` contains '!',
    # `log_each_step` becomes `False`
    log_each_step = True
    if '!' in exp_name:
        exp_name = exp_name.replace('!', '')
        log_each_step = False

    # if `exp_name` contains a '@' character,
    # the number following '@' is considered as
    # the desired random seed for the experiment
    split = exp_name.split('@')
    if len(split) == 2:
        seed = int(split[1])
        exp_name = split[0]
    # seed = 9427
    cnf = Conf(conf_file_path=conf_file_path, seed=seed, exp_name=exp_name, log=log_each_step)
    """@nni.variable(nni.choice(32,64,128), name=batch_size)"""
    batch_size = 128
    """@nni.variable(nni.choice(0.1, 0.3, 0.5), name=drop_rate)"""
    drop_rate = 0.1
    """@nni.variable(nni.choice(80, 160, 240), name=hidden_size)"""
    hidden_size = 240
    """@nni.variable(nni.choice(0.0001, 0.00001), name=learning_rate)"""
    learning_rate = 0.001

    cnf.lr = learning_rate
    cnf.batch_size = batch_size
    cnf.all_params['batch_size'] = batch_size
    cnf.all_params['lr'] = learning_rate
    cnf.all_params['learning_rate'] = learning_rate
    cnf.all_params['hidden_layer_size'] = hidden_size
    cnf.all_params['dropout_rate'] = drop_rate

    print(f'\n{cnf}')

    print(f'\n▶ Starting Experiment \'{exp_name}\' [seed: {cnf.seed}]')

    if inference:
        ts_model = TS(cnf=cnf)
        # ts_model.resume(ts_model.model, ts_model.optimizer, "/home/cyf/Downloads/28model/epoch_2")
        ts_model.test()
    else:
        trainer = Trainer(cnf=cnf)
        # trainer.resume(trainer.model, trainer.optimizer, "/home/cyf/Downloads/28model/epoch_2")
        trainer.run()


if __name__ == '__main__':
    main()
