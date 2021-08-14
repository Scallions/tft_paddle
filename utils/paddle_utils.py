import os
import shutil
import paddle
from collections import deque
from utils.io_utils import create_dir
from utils import log_utils as logger


def save_model(model,optimizer,save_dir,epoch,iters):
    create_dir(save_dir)
    current_save_dir = os.path.join(save_dir,"epoch_{}_iter_{}".format(epoch,iters))
    nranks = paddle.distributed.ParallelEnv().nranks
    create_dir(current_save_dir)

    if nranks > 1:
        paddle.distributed.fleet.init(is_collective=True)
        optimizer = paddle.distributed.fleet.distributed_optimizer(
            optimizer)  # The return is Fleet object
        model = paddle.distributed.fleet.distributed_model(model)

    paddle.save(model.state_dict(),
                os.path.join(current_save_dir, 'model.pdparams'))
    paddle.save(optimizer.state_dict(),
                os.path.join(current_save_dir, 'model.pdopt'))




def resume(model, optimizer, resume_model):
    if resume_model is not None:
        logger.info('Resume model from {}'.format(resume_model))
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
        logger.info('No model needed to resume.')

        
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self,save_dir, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_dir = save_dir
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_model(model,optimizer,self.save_dir,self.epoch,self.iters)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_model(model,optimizer,self.save_dir,self.epoch,self.iters)	
            self.counter = 0

    def save_checkpoint(self, val_loss, model,epoch,iters):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        save_model(model,optimizer,self.save_dir,epoch,iters)	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss
