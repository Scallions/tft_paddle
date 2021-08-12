

configs = {}
## 网络设置
configs['time_varying_categoical_variables'] = 1
configs['time_varying_real_variables_encoder'] = 4
configs['time_varying_real_variables_decoder'] = 3
configs['num_masked_series'] = 1
### embed
configs['static_embedding_vocab_sizes'] = [369]
configs['time_varying_embedding_vocab_sizes'] = [369]
configs['embedding_dim'] = 8
### lstm
configs['lstm_hidden_dimension'] = 160
configs['lstm_layers'] = 1
configs['dropout'] = 0.05
### attention
configs['attn_heads'] = 4
### qloss
configs['num_quantiles'] = 3
configs['vailid_quantiles'] = [0.1, 0.5, 0.9]
## 训练设置
configs['device'] = 'cpu'
configs['epochs'] = 100
configs['learning_rate'] = 0.001
## 数据集设置
configs['data_csv_path'] = 'dataset/LD2011_2014.csv'
configs['batch_size'] = 64
configs['encode_length'] = 168 # 输入长度
configs['seq_length'] = 192 # 输入加预测长度
configs['id_col'] = 'categorical_id'
configs['time_col'] ='hours_from_start'
configs['static_cols'] = ['categorical_id']
configs['static_variables'] = len(configs['static_cols'])
configs['input_cols'] =['power_usage', 'hour', 'day_of_week', 'hours_from_start', 'categorical_id']
configs['target_col'] = 'power_usage'
configs['max_samples'] = 1000

## 输入输出设置和数据集对应
configs['input_size'] = 5
configs['output_size'] = 1