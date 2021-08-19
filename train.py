from data.dataloader import load_data
import pandas as pd
import paddle
import numpy as np
import utils
import model
import data
import config
from utils import log_utils as logger
from utils import paddle_utils,io_utils
from tqdm import tqdm
pd.set_option('max_columns', 1000)

## seed
seed = 2512
paddle.seed(seed)
np.random.seed(seed)
import random
random.seed(seed)

## load configs
configs = config.configs
val_configs = config.val_configs

## nni set
"""@nni.variable(nni.choice(16,32,64, 128), name=batch_size)"""
batch_size = 64
"""@nni.variable(nni.choice(0.1, 0.3, 0.5), name=drop_rate)"""
drop_rate = 0.1
"""@nni.variable(nni.choice(40, 80, 160, 240), name=hidden_size)"""
hidden_size = 160
"""@nni.variable(nni.choice(0.0001, 0.001), name=learning_rate)"""
learning_rate = 0.001

configs['batch_size'] = batch_size
configs['dropout'] = drop_rate
configs['lstm_hidden_dimension'] = hidden_size
configs['learning_rate'] = learning_rate

## train
### load model
tft_model = model.create_model(configs)
### load optimizer
optimizer = utils.utils.create_optimizer(configs, tft_model)

clip = paddle.nn.ClipGradByNorm(clip_norm=0.01)
# scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.001, step_size=2, gamma=0.5, verbose=False)
optimizer = paddle.optimizer.Adam(
        learning_rate=configs['learning_rate'],
        # learning_rate = scheduler,
        parameters=tft_model.parameters(),
        grad_clip = clip,
        )
### load dataloader
train, val, test, dataformer = data.load_data(configs)
train_loader = data.create_dataloader(configs, train)
val_loader = data.create_dataloader(val_configs, val)
### loss func
q_loss_func = model.QuantileLoss([0.1, 0.5, 0.9])
q_90_loss_func = model.QuantileLoss([0.9])

log_step = len(train_loader) // 20
tft_model.train()
for epoch in range(configs["epochs"]):
    epoch_loss = [] # record loss
    pbar = tqdm(enumerate(train_loader))
    v_loss = 1
    m_v_loss = 1
    for iter, batch in pbar:
        # output, encoder_output, decoder_putput, attn, attn_weights, _, _ = tft_model(batch)
        output = tft_model(batch)
        loss = q_loss_func(output[:,:,:].reshape((-1,3)), batch['outputs'][:,:,0].flatten().astype('float32'))
        if loss.isnan().any().item() == True:
            exit()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        msg = "epoch:{}/{},iter:{}/{},train_loss:{:.4f}".format(epoch + 1, configs["epochs"],
                                                                                iter + 1,
                                                                                len(train_loader), loss.item())
        epoch_loss.append(loss.item())
        if iter % log_step == 0:
            io_utils.write_log(msg, 'experiment/log', 'tft_model')
        if iter % (4*log_step) == 0:
        # if iter % 1 == 0:
            tft_model.eval()
            val_losses = []
            with paddle.no_grad():
                for val_batch in val_loader:
                    # val_output, val_encoder_output, val_decoder_putput, val_attn, val_attn_weights, _, _ = tft_model(val_batch)
                    val_output = tft_model(val_batch)
                    target = utils.utils.unnormalize_tensor(dataformer, val_batch['outputs'].squeeze(), val_batch['identifier'][0][0])
                    p90_forecast = utils.utils.unnormalize_tensor(dataformer, val_output[:,:,2], val_batch['identifier'][0][0])
                    # val_loss = q_90_loss_func(val_output[:,:,:].reshape((-1,3))[:,2:], val_batch['outputs'][:,:,0].flatten().astype('float32')) / batch['outputs'].abs().mean()
                    val_losses.append(utils.utils.numpy_normalised_quantile_loss(target, p90_forecast,0.9))
                    # break
            v_loss = np.mean(val_losses)
            m_v_loss = min(v_loss, m_v_loss)
            # logger.info(msg)
            # pbar.set_postfix(str=msg)
            """@nni.report_intermediate_result(v_loss)"""
            io_utils.write_log(msg + ",val_loss:{:.4f},m_loss:{:.4f}".format(v_loss, m_v_loss), 'experiment/log', 'tft_model')
            tft_model.train()
            # scheduler.step()
        msg = msg + ",val_loss:{:.4f},m_loss:{:.4f}".format(v_loss, m_v_loss)                                                     
        # logger.info(msg)
        pbar.set_postfix(str=msg)
        if (iter+1) % len(train_loader) == 0:
            paddle_utils.save_model(tft_model,optimizer,
                                    'experiment/save_model',epoch+1,iter+1)
    """@nni.report_final_result(m_v_loss)"""
    print(f"epoch_loss:{np.mean(epoch_loss):.4f}")


## output results