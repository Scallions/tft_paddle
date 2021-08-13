import math
import paddle
import paddle.nn as nn
# import paddle.nn.functional as F
# from paddle.io import DataLoader,Dataset
# from paddle.vision import transforms

class QuantileLoss(nn.Layer):
    def __init__(self,quantiles):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert target.stop_gradient
        assert preds.shape[0] == target.shape[0]
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:,i]
            left = (q-1) * errors 
            right = q * errors 
            losses.append(paddle.maximum(left, right).unsqueeze(1))
        loss = paddle.mean(paddle.sum(paddle.concat(losses,axis=1),axis=1))
        return loss


class Embedding(nn.Layer):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, sparse=False, weight_attr=None, name=None):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx, sparse, weight_attr, name)
    
    def forward(self, x):
        return self.emb(x.astype('int64'))


class LSTM(nn.Layer):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    def forward(self, x, hs):
        # x: b, t, l
        x = paddle.transpose(x, [1,0,2])
        x, (h, c) = self.lstm(x, hs)
        x = paddle.transpose(x, [1,0,2])
        return x, (h, c)


class TimeDistributed(nn.Layer):
    ## Takes any module and stacks the time dimension with the batch dimenison of inputs before apply the module
    ## From: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.shape) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.reshape((-1,x.shape[-1]))

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.reshape((x.shape[0],-1,y.shape[-1]))
        else:
            y = y.reshape((-1,x.shape[1],y.shape[-1]))
        return y


class GLU(nn.Layer):
    # Gated Linear Unit
    def __init__(self, input_size):
        super(GLU, self).__init__()

        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return paddle.multiply(sig, x)


class GatedResidualNetwork(nn.Layer):
    def __init__(self, input_size, hidden_state_size, output_size, dropout, hidden_context_size=None,
                 batch_first=False):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_context_size = hidden_context_size
        self.hidden_state_size = hidden_state_size
        self.dropout = dropout

        if self.input_size != self.output_size:
            self.skip_layer = TimeDistributed(nn.Linear(self.input_size, self.output_size))

        self.fc1 = TimeDistributed(nn.Linear(self.input_size, self.hidden_state_size), batch_first=batch_first)
        self.elu1 = nn.ELU()

        if self.hidden_context_size is not None:
            self.context = TimeDistributed(nn.Linear(self.hidden_context_size, self.hidden_state_size),
                                           batch_first=batch_first)

        self.fc2 = TimeDistributed(nn.Linear(self.hidden_state_size, self.output_size), batch_first=batch_first)
        self.elu2 = nn.ELU()

        self.dropout = nn.Dropout(self.dropout)
        self.bn = TimeDistributed(nn.BatchNorm1D(self.output_size), batch_first=batch_first)
        self.gate = TimeDistributed(GLU(self.output_size), batch_first=batch_first)

    def forward(self, x, context=None):

        if self.input_size != self.output_size:
            residual = self.skip_layer(x)
        else:
            residual = x

        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu1(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gate(x)
        x = x + residual
        x = self.bn(x)

        return x

class PositionalEncoder(nn.Layer):
    def __init__(self, d_model, max_seq_len=160):
        super().__init__()
        self.d_model = d_model
        pe = paddle.zeros((max_seq_len, d_model))
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with paddle.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.shape[0]
            pe = self.pe[:, :seq_len].reshape((seq_len,1,self.d_model))
            x = x + pe
            return x


class VariableSelectionNetwork(nn.Layer):
    def __init__(self, input_size, num_inputs, hidden_size, dropout, context=None):
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.dropout = dropout
        self.context = context

        if self.context is not None:
            self.flattened_grn = GatedResidualNetwork(self.num_inputs * self.input_size, self.hidden_size,
                                                      self.num_inputs, self.dropout, self.context)
        else:
            self.flattened_grn = GatedResidualNetwork(self.num_inputs * self.input_size, self.hidden_size,
                                                      self.num_inputs, self.dropout)

        self.single_variable_grns = nn.LayerList()
        for i in range(self.num_inputs):
            self.single_variable_grns.append(
                GatedResidualNetwork(self.input_size, self.hidden_size, self.hidden_size, self.dropout))

        self.softmax = nn.Softmax()

    def forward(self, embedding, context=None):
        if context is not None:
            sparse_weights = self.flattened_grn(embedding, context)
        else:
            sparse_weights = self.flattened_grn(embedding)

        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)

        var_outputs = []
        for i in range(self.num_inputs):
            ##select slice of embedding belonging to a single input
            var_outputs.append(
                self.single_variable_grns[i](embedding[:, :, (i * self.input_size): (i + 1) * self.input_size]))

        var_outputs = paddle.stack(var_outputs, axis=-1)

        outputs = var_outputs * sparse_weights

        outputs = outputs.sum(axis=-1)

        return outputs, sparse_weights


class TFT(nn.Layer):
    def __init__(self, config):
        super(TFT, self).__init__()
        self.batch_size = config['batch_size']
        self.static_variables = config['static_variables']
        self.encode_length = config['encode_length']
        self.time_varying_categoical_variables = config['time_varying_categoical_variables']
        self.time_varying_real_variables_encoder = config['time_varying_real_variables_encoder']
        self.time_varying_real_variables_decoder = config['time_varying_real_variables_decoder']
        self.num_input_series_to_mask = config['num_masked_series']
        self.hidden_size = config['lstm_hidden_dimension']
        self.lstm_layers = config['lstm_layers']
        self.dropout = config['dropout']
        self.embedding_dim = config['embedding_dim']
        self.attn_heads = config['attn_heads']
        self.num_quantiles = config['num_quantiles']
        self.valid_quantiles = config['vailid_quantiles']
        self.seq_length = config['seq_length']

        self.static_embedding_layers = nn.LayerList()
        for i in range(self.static_variables):
            emb = Embedding(config['static_embedding_vocab_sizes'][i], config['embedding_dim'])
            self.static_embedding_layers.append(emb)

        self.time_varying_embedding_layers = nn.LayerList()
        for i in range(self.time_varying_categoical_variables):
            emb = TimeDistributed(
                Embedding(config['time_varying_embedding_vocab_sizes'][i], config['embedding_dim']),
                batch_first=True)
            self.time_varying_embedding_layers.append(emb)

        self.time_varying_linear_layers = nn.LayerList()
        for i in range(self.time_varying_real_variables_encoder):
            emb = TimeDistributed(nn.Linear(1, config['embedding_dim']), batch_first=True)
            self.time_varying_linear_layers.append(emb)

        self.encoder_variable_selection = VariableSelectionNetwork(config['embedding_dim'],
                                                                   (config['time_varying_real_variables_encoder'] +
                                                                    config['time_varying_categoical_variables']),
                                                                   self.hidden_size,
                                                                   self.dropout,
                                                                   config['embedding_dim'] * config['static_variables'])

        self.decoder_variable_selection = VariableSelectionNetwork(config['embedding_dim'],
                                                                   (config['time_varying_real_variables_decoder'] +
                                                                    config['time_varying_categoical_variables']),
                                                                   self.hidden_size,
                                                                   self.dropout,
                                                                   config['embedding_dim'] * config['static_variables'])

        self.lstm_encoder_input_size = config['embedding_dim'] * (config['time_varying_real_variables_encoder'] +
                                                                  config['time_varying_categoical_variables'] +
                                                                  config['static_variables'])

        self.lstm_decoder_input_size = config['embedding_dim'] * (config['time_varying_real_variables_decoder'] +
                                                                  config['time_varying_categoical_variables'] +
                                                                  config['static_variables'])

        self.lstm_encoder = LSTM(input_size=self.hidden_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.lstm_layers,
                                    dropout=config['dropout'])

        self.lstm_decoder = LSTM(input_size=self.hidden_size,
                                    hidden_size=self.hidden_size,
                                    num_layers=self.lstm_layers,
                                    dropout=config['dropout'])

        self.post_lstm_gate = TimeDistributed(GLU(self.hidden_size))
        self.post_lstm_norm = TimeDistributed(nn.BatchNorm1D(self.hidden_size))

        self.static_enrichment = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size,
                                                      self.dropout, config['embedding_dim'] * self.static_variables)

        self.position_encoding = PositionalEncoder(self.hidden_size, self.seq_length)

        self.multihead_attn = nn.MultiHeadAttention(self.hidden_size, self.attn_heads, need_weights=True)
        self.post_attn_gate = TimeDistributed(GLU(self.hidden_size))

        self.post_attn_norm = TimeDistributed(nn.BatchNorm1D(self.hidden_size))
        self.pos_wise_ff = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout)

        self.pre_output_norm = TimeDistributed(nn.BatchNorm1D(self.hidden_size))
        self.pre_output_gate = TimeDistributed(GLU(self.hidden_size))

        self.output_layer = TimeDistributed(nn.Linear(self.hidden_size, self.num_quantiles), batch_first=True)

    def init_hidden(self):
        return paddle.zeros((self.lstm_layers, self.batch_size, self.hidden_size))

    def apply_embedding(self, x, static_embedding, apply_masking):
        ###x should have dimensions (batch_size, timesteps, input_size)
        ## Apply masking is used to mask variables that should not be accessed after the encoding steps
        # Time-varying real embeddings
        if apply_masking:
            time_varying_real_vectors = []
            for i in range(self.time_varying_real_variables_decoder):
                emb = self.time_varying_linear_layers[i + self.num_input_series_to_mask](
                    x[:, :, i + self.num_input_series_to_mask].reshape((x.shape[0], -1, 1)))
                time_varying_real_vectors.append(emb)
            time_varying_real_embedding = paddle.concat(time_varying_real_vectors, axis=2)

        else:
            time_varying_real_vectors = []
            for i in range(self.time_varying_real_variables_encoder):
                emb = self.time_varying_linear_layers[i](x[:, :, i].reshape((x.shape[0], -1, 1)))
                time_varying_real_vectors.append(emb)
            time_varying_real_embedding = paddle.concat(time_varying_real_vectors, axis=2)

        ##Time-varying categorical embeddings (ie hour)
        time_varying_categoical_vectors = []
        for i in range(self.time_varying_categoical_variables):
            emb = self.time_varying_embedding_layers[i](
                x[:, :, self.time_varying_real_variables_encoder + i].reshape((x.shape[0], -1, 1)))
            time_varying_categoical_vectors.append(emb)
        time_varying_categoical_embedding = paddle.concat(time_varying_categoical_vectors, axis=2)

        ##repeat static_embedding for all timesteps
        static_embedding = paddle.concat(time_varying_categoical_embedding.shape[1] * [static_embedding])
        static_embedding = static_embedding.reshape((time_varying_categoical_embedding.shape[0],
                                                 time_varying_categoical_embedding.shape[1], -1))

        ##concatenate all embeddings
        embeddings = paddle.concat([static_embedding, time_varying_categoical_embedding, time_varying_real_embedding],
                               axis=2)

        return embeddings.reshape((-1, x.shape[0], embeddings.shape[2]))

    def encode(self, x, hidden=None):

        if hidden is None:
            hidden = self.init_hidden()

        output, (hidden, cell) = self.lstm_encoder(x, (hidden, hidden))

        return output, hidden

    def decode(self, x, hidden=None):

        if hidden is None:
            hidden = self.init_hidden()

        output, (hidden, cell) = self.lstm_decoder(x, (hidden, hidden))

        return output, hidden

    def forward(self, x):

        ##inputs should be in this order
        # static
        # time_varying_categorical
        # time_varying_real

        embedding_vectors = []
        for i in range(self.static_variables):
            # only need static variable from the first timestep
            emb = self.static_embedding_layers[i](x['identifier'][:, 0, i])
            embedding_vectors.append(emb)

        ##Embedding and variable selection
        static_embedding = paddle.concat(embedding_vectors, axis=1)
        embeddings_encoder = self.apply_embedding(x['inputs'][:, :self.encode_length, :],
                                                  static_embedding, apply_masking=False)
        embeddings_decoder = self.apply_embedding(x['inputs'][:, self.encode_length:, :],
                                                  static_embedding, apply_masking=True)
        embeddings_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings_encoder[:, :, :-(self.embedding_dim * self.static_variables)],
            embeddings_encoder[:, :, -(self.embedding_dim * self.static_variables):])
        embeddings_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            embeddings_decoder[:, :, :-(self.embedding_dim * self.static_variables)],
            embeddings_decoder[:, :, -(self.embedding_dim * self.static_variables):])

        pe = self.position_encoding(paddle.zeros((self.seq_length, 1, embeddings_encoder.shape[2])))

        embeddings_encoder = embeddings_encoder + pe[:self.encode_length, :, :]
        embeddings_decoder = embeddings_decoder + pe[self.encode_length:, :, :]

        ##LSTM
        lstm_input = paddle.concat([embeddings_encoder, embeddings_decoder], axis=0)
        encoder_output, hidden = self.encode(embeddings_encoder)
        decoder_output, _ = self.decode(embeddings_decoder, hidden)
        lstm_output = paddle.concat([encoder_output, decoder_output], axis=0)

        ##skip connection over lstm
        lstm_output = self.post_lstm_gate(lstm_output + lstm_input)

        ##static enrichment
        static_embedding = paddle.concat(lstm_output.shape[0] * [static_embedding]).reshape((lstm_output.shape[0],
                                                                                    lstm_output.shape[1], -1))
        attn_input = self.static_enrichment(lstm_output, static_embedding)

        ##skip connection over lstm
        attn_input = self.post_lstm_norm(lstm_output)

        # attn_input = self.position_encoding(attn_input)

        ##Attention
        ## Pytorch Q(L,N,E) but in paddle Q(N,L,E)
        attn_input = paddle.transpose(attn_input, [1,0,2])
        attn_output, attn_output_weights = self.multihead_attn(attn_input[:, self.encode_length:, :],
                                                               attn_input[:, :self.encode_length:, :],
                                                               attn_input[:, :self.encode_length, :])
        attn_output = paddle.transpose(attn_output, [1,0,2])
        attn_input = paddle.transpose(attn_input, [1,0,2])
        attn_output_weights = paddle.transpose(attn_output_weights, [0,1,3,2])
        ##skip connection over attention
        attn_output = self.post_attn_gate(attn_output) + attn_input[self.encode_length:, :, :]
        attn_output = self.post_attn_norm(attn_output)

        output = self.pos_wise_ff(attn_output)  # [self.encode_length:,:,:])

        ##skip connection over Decoder
        output = self.pre_output_gate(output) + lstm_output[self.encode_length:, :, :]

        # Final output layers
        output = self.pre_output_norm(output)
        output = self.output_layer(output.reshape((self.batch_size, -1, self.hidden_size)))

        return output, encoder_output, decoder_output, attn_output, attn_output_weights, encoder_sparse_weights, decoder_sparse_weights


def create_model(configs):
    from model.tft_model import TFT
    return TFT(config=configs)

if __name__ == '__main__':
    static_cols = ['meter']
    categorical_cols = ['hour']
    real_cols = ['power_usage', 'hour', 'day']
    config = {}
    config['static_variables'] = len(static_cols)
    config['time_varying_categoical_variables'] = 1
    config['time_varying_real_variables_encoder'] = 4
    config['time_varying_real_variables_decoder'] = 3
    config['num_masked_series'] = 1
    config['static_embedding_vocab_sizes'] = [369]
    config['time_varying_embedding_vocab_sizes'] = [369]
    config['embedding_dim'] = 8
    config['lstm_hidden_dimension'] = 160
    config['lstm_layers'] = 1
    config['dropout'] = 0.05
    config['batch_size'] = 64
    config['encode_length'] = 168
    config['attn_heads'] = 4
    config['num_quantiles'] = 3
    config['vailid_quantiles'] = [0.1, 0.5, 0.9]