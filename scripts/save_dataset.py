import sys 
sys.path.append(".")

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

import pickle

pd.set_option('max_columns', 1000)

## load configs
configs = config.configs
val_configs = config.val_configs

### load dataloader
train, val, test = data.load_data(configs)

train = data.create_dataset(configs, train)
val = data.create_dataset(val_configs, val)
test = data.create_dataset(val_configs, test)


train
with open('dataset/train.pkl', 'wb') as f:
    pickle.dump(train,f)

# val
with open('dataset/val.pkl', 'wb') as f:
    pickle.dump(val,f)

# test
with open('dataset/test.pkl', 'wb') as f:
    pickle.dump(test,f)