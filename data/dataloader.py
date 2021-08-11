from paddle.io import Dataset, DataLoader
import numpy as np
import pandas as pd
from data.electricity import create_dataformer

class TSDataset(Dataset):
    def __init__(self,id_col, static_cols, time_col, input_cols,
                 target_col, time_steps, max_samples,
                 input_size, num_encoder_steps,num_static,
                 output_size, data):
        super().__init__()
        self.time_steps = time_steps
        self.input_size = input_size
        self.output_size = output_size
        self.num_encoder_steps = num_encoder_steps
        
        
        data.sort_values(by=[id_col, time_col], inplace=True)
        print('Getting valid sampling locations.')
        
        valid_sampling_locations = []
        split_data_map = {}
        for identifier, df in data.groupby(id_col):
            num_entries = len(df)
            if num_entries >= self.time_steps:
                valid_sampling_locations += [
                    (identifier, self.time_steps + i)
                    for i in range(num_entries - self.time_steps + 1)
                ]
            split_data_map[identifier] = df

        self.inputs = np.zeros((max_samples, self.time_steps, self.input_size))
        self.outputs = np.zeros((max_samples, self.time_steps, self.output_size))
        self.time = np.empty((max_samples, self.time_steps, 1))
        self.identifiers = np.empty((max_samples,self.time_steps, num_static))

        if max_samples > 0 and len(valid_sampling_locations) > max_samples:
            print('Extracting {} samples...'.format(max_samples))
            ranges = [valid_sampling_locations[i] for i in np.random.choice(
                  len(valid_sampling_locations), max_samples, replace=False)]
        else:
            print('Max samples={} exceeds # available segments={}'.format(
                  max_samples, len(valid_sampling_locations)))
            ranges = valid_sampling_locations
        
        for i, tup in enumerate(ranges):
            if ((i + 1) % 10000) == 0:
                print(i + 1, 'of', max_samples, 'samples done...')
            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx -
                                               self.time_steps:start_idx]
            self.inputs[i, :, :] = sliced[input_cols]
            self.outputs[i, :, :] = sliced[[target_col]]
            self.time[i, :, 0] = sliced[time_col]
            self.identifiers[i,:, :] = sliced[static_cols]

        self.sampled_data = {
            'inputs': self.inputs,
            'outputs': self.outputs[:, self.num_encoder_steps:, :],
            'active_entries': np.ones_like(self.outputs[:, self.num_encoder_steps:, :]),
            'time': self.time,
            'identifier': self.identifiers
        }

    
    def __getitem__(self, index):
        s = {
        'inputs': self.inputs[index].astype('float32'),
        'outputs': self.outputs[index, self.num_encoder_steps:, :].astype('float32'),
        'active_entries': np.ones_like(self.outputs[index, self.num_encoder_steps:, :]),
        'time': self.time[index],
        'identifier': self.identifiers[index].astype('float32')
        }
        return s

    def __len__(self):
        return self.inputs.shape[0]

def load_data(configs):
    data_csv_path = "dataset/LD2011_2014.csv"

    raw_data = pd.read_csv(data_csv_path, index_col=0)
    # print(raw_data.head())
    dataformer = create_dataformer()
    train, valid, test = dataformer.split_data(raw_data)
    return train, valid, test

def create_dataset(configs, data):
    # process configs
    id_col = 'categorical_id'
    time_col='hours_from_start'
    static_cols = ['categorical_id']
    num_static = 1
    input_cols =['power_usage', 'hour', 'day_of_week', 'hours_from_start', 'categorical_id']
    target_col = 'power_usage'
    time_steps=configs['seq_length']
    num_encoder_steps = 168
    output_size = 1
    max_samples = 1000
    input_size = 5

    return TSDataset(id_col, static_cols, time_col, input_cols,
                      target_col, time_steps, max_samples,
                     input_size, num_encoder_steps, num_static, output_size, data)

def create_dataloader(configs, data):
    # process configs

    # create dataset
    dataset = create_dataset(configs, data)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=configs["batch_size"],
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    return dataloader