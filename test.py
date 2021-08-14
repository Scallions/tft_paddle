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
import matplotlib.pyplot as plt

pd.set_option('max_columns', 1000)

## load configs
configs = config.configs
val_configs = config.val_configs

## train
### load model
tft_model = model.create_model(configs)
optimizer = utils.utils.create_optimizer(configs, tft_model)
paddle_utils.resume(tft_model, optimizer,"experiment/save_model/epoch_10_iter_7812/")
### load dataloader
train, val, test = data.load_data(configs)
val_loader = data.create_dataloader(val_configs, val)
### loss func
q_90_loss_func = model.QuantileLoss([0.9])


tft_model.eval()
val_losses = []
with paddle.no_grad():
    for val_batch in val_loader:
        # val_output, val_encoder_output, val_decoder_putput, val_attn, val_attn_weights, _, _ = tft_model(val_batch)
        val_output = tft_model(val_batch)
        val_loss = q_90_loss_func(val_output[:,:,:].reshape((-1,3))[:,2:], val_batch['outputs'][:,:,0].flatten().astype('float32'))
        val_losses.append(val_loss.item())
plt.plot(range(168), val_batch['inputs'][0,:168,0].numpy())
plt.plot(range(168,192), val_batch['outputs'][0,:,:].numpy(), label='true')
plt.plot(range(168,192), val_output[0,:,:].numpy())
plt.legend()
plt.savefig('dataset/test.png')
msg = "[EVAL] val_loss: {:4f}".format(np.mean(val_losses))
logger.info(msg)




## output results