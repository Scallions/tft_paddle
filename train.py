from data.dataloader import load_data
import pandas as pd
import numpy as np
import utils
import model
import data
from tqdm import tqdm
from loguru import logger
import config

pd.set_option('max_columns', 1000)

## load configs
configs = config.configs

## train
### load model
tft_model = model.create_model(configs)
### load optimizer
optimizer = utils.utils.create_optimizer(configs, tft_model)
### load dataloader
train, val, test = data.load_data(configs)
dataloader = data.create_dataloader(configs, train)
val_loader = data.create_dataloader(configs, val)
### loss func
q_loss_func = model.QuantileLoss(configs['vailid_quantiles'])
q_90_loss_func = model.QuantileLoss([0.9])
print(len(dataloader))
### train step
#### epoch
losses = []
#pbar = tqdm(range(configs["epochs"]))
for epoch in range(configs["epochs"]):
    epoch_loss = [] # record loss
    ##### batch
    for iter, batch in enumerate(dataloader):
        # output, encoder_ouput, decoder_output, attn, attn_weights = tft_model(batch)
        output, encoder_output, decoder_putput, attn, attn_weights, _, _ = tft_model(batch)
        loss = q_loss_func(output[:,:,:].reshape((-1,3)), batch['outputs'][:,:,0].flatten().astype('float32'))
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        epoch_loss.append(loss.item())
        if (iter+1) % 5 == 0:
            for batch in val_loader:
                output, encoder_output, decoder_putput, attn, attn_weights, _, _ = tft_model(batch)
                loss_ = q_90_loss_func(output[:,:,:].reshape((-1,3)), batch['outputs'][:,:,0].flatten().astype('float32'))
                break
            print(f"epoch:{epoch+1} \t iter: {iter+1} \t loss:{loss.item()} \t val_loss:{loss_.item()}")
    losses.append(np.mean(epoch_loss))


## output results
