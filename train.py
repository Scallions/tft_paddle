# import lib
from data.dataloader import load_data
import pandas as pd
import numpy as np
import utils
import model
import data
from tqdm import tqdm
from loguru import logger

pd.set_option('max_columns', 1000)

## load configs
static_cols = ['categorical_id']
configs = {}
configs['static_variables'] = len(static_cols)
configs['time_varying_categoical_variables'] = 1
configs['time_varying_real_variables_encoder'] = 4
configs['time_varying_real_variables_decoder'] = 3
configs['num_masked_series'] = 1
configs['static_embedding_vocab_sizes'] = [369]
configs['time_varying_embedding_vocab_sizes'] = [369]
configs['embedding_dim'] = 8
configs['lstm_hidden_dimension'] = 160
configs['lstm_layers'] = 1
configs['dropout'] = 0.05
configs['device'] = 'cpu'
configs['batch_size'] = 64
configs['encode_length'] = 168
configs['attn_heads'] = 4
configs['num_quantiles'] = 3
configs['vailid_quantiles'] = [0.1,0.5,0.9]
configs['seq_length'] = 192
configs['epochs'] = 10


## train
### load model
tft_model = model.create_model(configs) 
### load optimizer
optimizer = utils.utils.create_optimizer(configs, tft_model)
### load dataloader
train, val, test = data.load_data(configs)
dataloader = data.create_dataloader(configs, train)
### loss func
q_loss_func = model.QuantileLoss([0.1,0.5,0.9])

### train step
#### epoch
losses = []
pbar = tqdm(range(configs["epochs"]))
for epoch in pbar:
    epoch_loss = [] # record loss
    ##### batch
    for batch in dataloader:
        output, encoder_ouput, decoder_output, attn, attn_weights = tft_model(batch)
        loss= q_loss_func(output[:,:,:].view(-1,3), batch['outputs'][:,:,0].flatten().float())
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
    losses.append(np.mean(epoch_loss))
    pbar.set_description(f"epoch: {epoch} loss: {np.mean(epoch_loss)}")

    

## output results