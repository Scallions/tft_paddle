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
pd.set_option('max_columns', 1000)

## load configs
configs = config.configs
val_configs = config.valconfigs

## train
### load model
tft_model = model.create_model(configs)
### load optimizer
optimizer = utils.utils.create_optimizer(configs, tft_model)
### load dataloader
train, val, test = data.load_data(configs)
train_loader = data.create_dataloader(configs, train)
val_loader = data.create_dataloader(configs, val)
### loss func
q_loss_func = model.QuantileLoss([0.1, 0.5, 0.9])
q_90_loss_func = model.QuantileLoss([0.9])


tft_model.train()
for epoch in range(configs["epochs"]):
    epoch_loss = [] # record loss
    for iter, batch in enumerate(train_loader):
        # output, encoder_output, decoder_putput, attn, attn_weights, _, _ = tft_model(batch)
        output = tft_model(batch)
        loss = q_loss_func(output[:,:,:].reshape((-1,3)), batch['outputs'][:,:,0].flatten().astype('float32'))
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        if (iter+1) % 100 == 0:
            msg = "[TRAIN] epoch: {}/{}, iter: {}/{},\t train_loss: {:4f}".format(epoch + 1, configs["epochs"],
                                                                                  iter + 1,
                                                                                  len(train_loader), loss.item())
            logger.info(msg)
            io_utils.write_log(msg, 'experiment/log', 'tft_model')
        if (iter+1) % len(train_loader) == 0:
            tft_model.eval()
            val_losses = []
            with paddle.no_grad():
                for val_batch in val_loader:
                    # val_output, val_encoder_output, val_decoder_putput, val_attn, val_attn_weights, _, _ = tft_model(val_batch)
                    val_output = tft_model(val_batch)
                    val_loss = q_90_loss_func(val_output[:,:,:].reshape((-1,3))[:,2:], val_batch['outputs'][:,:,0].flatten().astype('float32')) / batch['outputs'].abs().mean()
                    val_losses.append(val_loss.item())
            msg = "[EVAL] val_loss: {:4f}".format(np.mean(val_losses))
            logger.info(msg)
            io_utils.write_log(msg, 'experiment/log', 'tft_model')
        tft_model.train()
        if (iter+1) % len(train_loader) == 0:
            paddle_utils.save_model(tft_model,optimizer,
                                    'experiment/save_model',epoch+1,iter+1)




## output results