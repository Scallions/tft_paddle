import os
import shutil
import paddle
from collections import deque
from utils.io_utils import create_dir
from utils import log_utils as logger


def save_model(model,optimizer,save_dir,epoch,iter,keep_checkpoint_max=5):
    create_dir(save_dir)
    current_save_dir = os.path.join(save_dir,"epoch_{}_iter_{}".format(epoch,iter))
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

    save_models = deque()
    save_models.append(current_save_dir)
    # if len(save_models) > keep_checkpoint_max > 0:
    #     model_to_remove = save_models.popleft()
    #     shutil.rmtree(model_to_remove)

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