# import lib
import pandas as pd
import numpy as np
import utils
import model
import data
from tqdm import tqdm
from loguru import logger

pd.set_option('max_columns', 1000)

## load configs
configs = {}


## train
### load model
tft_model = model.create_model(configs) 
### load dataloader
dataloader = data.create_dataloader(configs)
### load optimizer
optimizer = utils.create_optimizer(configs)

### train step
#### epoch
losses = []
for epoch in tqdm(configs.epochs):
	epoch_loss = [] # record loss
	##### batch
	for batch in dataloader:
		output, encoder_ouput, decoder_output, attn, attn_weights = model(batch)
        loss= utils.q_loss_func(output[:,:,:].view(-1,3), batch['outputs'][:,:,0].flatten().float())
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
	losses.append(np.mean(epoch_loss))
	logger.info(f"epoch: {epoch} loss: {np.mean(epoch_loss)}")
	

## output results