from model.model import Embedding
import paddle
from paddle.fluid.layers.nn import embedding
import paddle.nn as nn


## quantileloss
class QuantileLoss(nn.Layer):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert target.stop_gradient
        assert preds.shape[0] == target.shape[0]
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:,i]
            left = (1-q) * nn.functional.relu(-errors)
            right = q * nn.functional.relu(errors) 
            losses.append((left + right).unsqueeze(1))
        loss = paddle.mean(paddle.sum(paddle.concat(losses,axis=1),axis=1))
        return loss

class TimeDistributed(nn.Layer):
    def __init__(self, module, batch_first=True):
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

class RealEmbedding(nn.Layer):
    def __init__(self, input_size, embedding_size):
        super().__init__()
        self.layer = TimeDistributed(nn.Layer(input_size, embedding_size))

    def forward(self, x):
        return self.layer(x)
        

class Linear(nn.Layer):
    def __init__(self, inp_size, out_size, use_td=False):
        super().__init__()
        self.linear = nn.Linear(inp_size, out_size)
        if use_td:
            self.linear = TimeDistributed(self.linear)

    def forward(self, x):
        return self.linear(x)

class GLU(nn.Layer):
    # Gated Linear Unit
    def __init__(self, inp_size, hidden_size, dropout=None, use_td=True):
        super().__init__()
        self.dropout = dropout

        if self.dropout is not None:
            self.dropout_layer = nn.Dropout(dropout)
        self.fc1 = Linear(inp_size, hidden_size, use_td)
        self.fc2 = Linear(inp_size, hidden_size, use_td)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout_layer(x)
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return paddle.multiply(sig, x)

class Add_Norm(nn.Layer):
    def __init__(self, size):
        super().__init__()
        # TODO: shape
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, x, skip):
        return self.layer_norm(x + skip)

class GRN(nn.Layer):
    """ 
    Gated Residual Network
    """
    def __init__(self, input_size, output_size, hidden_state_size, dropout, use_td=False, add_ctx=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_state_size = hidden_state_size
        self.dropout = dropout

        ## skip connection
        if self.input_size != self.output_size:
            self.skip_layer = TimeDistributed(nn.Linear(self.input_size, self.output_size))
        
        ## feedforward network
        self.hidden_layer = Linear(self.input_size, self.hidden_state_size, use_td)
        
        self.add_ctx = None
        if add_ctx is not None:
            self.add_ctx = Linear(hidden_state_size, hidden_state_size, use_td)

        self.elu = nn.ELU()
        self.hidden_layer2 = Linear(hidden_state_size, hidden_state_size, use_td)
        self.gating_layer = GLU(hidden_state_size, output_size, dropout, use_td)
        # TODO: shape       
        self.add_norm = Add_Norm(self.output_size)

    def forward(self, x, ctx=None):
        ## skip
        if self.input_size != self.output_size:
            skip = self.skip_layer(x)
        else:
            skip = x
        hidden = self.hidden_layer(x)
        if self.add_ctx is not None:
            hidden = hidden + self.add_ctx(ctx)
        hidden = self.elu(hidden)
        hidden = self.hidden_layer2(hidden)
        gating = self.gating_layer(hidden)
        return self.add_norm(gating, skip)

class StaticVariableSelectionNetwork(nn.Layer):
    def __init__(self, input_size, num_inputs, hidden_size, dropout, context=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.dropout = dropout
        self.context = context
        self.flatten = nn.Flatten()
        if self.context is not None:
            self.flattened_grn = GRN(self.num_inputs * self.input_size, self.num_inputs,
                                                      self.hidden_size, self.dropout, self.context)
        else:
            self.flattened_grn = GRN(self.num_inputs * self.input_size, self.num_inputs,
                                                      self.hidden_size, self.dropout)
        self.single_variable_grns = nn.LayerList()
        for i in range(self.num_inputs):
            self.single_variable_grns.append(
                GRN(self.input_size, self.hidden_size, self.hidden_size, self.dropout))
        self.softmax = nn.Softmax()

    def forward(self, embedding, context=None):
        flatten = self.flatten(embedding)
        if context is not None:
            mlp_outputs = self.flattened_grn(flatten, context)
        else:
            mlp_outputs = self.flattened_grn(flatten)
        sparse_weights = self.softmax(mlp_outputs).unsqueeze(2)

        trans_emb_list = []
        for i in range(self.num_inputs):
            trans_emb_list.append(
                self.single_variable_grns[i](embedding[:, :, (i * self.input_size): (i + 1) * self.input_size]))
        transformed_embbeding = paddle.concat(trans_emb_list, axis=1)
        combined = paddle.multiply(sparse_weights, transformed_embbeding)
        static_vec = paddle.sum(combined, axis=1)
        return static_vec, sparse_weights

class TimeVariableSelectionNetwork(nn.Layer):
    def __init__(self, time_steps, embedding_dim, input_nums, hidden_size, dropout, context=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_nums = input_nums
        self.time_steps = time_steps
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.context = context
        self.flatten = nn.Flatten()
        if self.context is not None:
            self.flattened_grn = GRN(self.embedding_dim * self.input_nums, self.input_nums,
                                                      self.hidden_size, self.dropout, self.context)
        else:
            self.flattened_grn = GRN(self.embedding_dim * self.input_nums, self.input_nums,
                                                      self.hidden_size, self.dropout)
        self.single_variable_grns = nn.LayerList()
        for i in range(self.input_nums):
            self.single_variable_grns.append(
                GRN(self.embedding_dim, self.hidden_size, self.hidden_size, self.dropout))
        self.softmax = nn.Softmax()

    def forward(self, embedding, context=None):
        # flatten = self.flatten(embedding)
        flatten = embedding.reshape([-1, self.time_steps, self.embedding_dim * self.input_nums])
        if context is not None:
            mlp_outputs = self.flattened_grn(flatten, context)
        else:
            mlp_outputs = self.flattened_grn(flatten)
        sparse_weights = self.softmax(mlp_outputs).unsqueeze(2)

        trans_emb_list = []
        for i in range(self.input_nums):
            trans_emb_list.append(
                self.single_variable_grns[i](embedding[:, :, :, i]))
        transformed_embbeding = paddle.stack(trans_emb_list, axis=-1)
        combined = paddle.multiply(sparse_weights, transformed_embbeding)
        static_vec = paddle.sum(combined, axis=-1)
        return static_vec, sparse_weights

class TFT(nn.Layer):
    def __init__(self, config):
        super().__init__()
        ## data params
        self.input_size = 5
        self.output_size = 1
        self.time_varying_categoical_variables = config['time_varying_categoical_variables']
        self.time_varying_real_variables_encoder = config['time_varying_real_variables_encoder']
        self.time_varying_real_variables_decoder = config['time_varying_real_variables_decoder']
        self.static_variables = config['static_variables']
        
        # relevant indices
        self._input_obs_loc = [0]
        self._static_input_loc = [4]
        self._known_regular_input_idx = [0]
        self._known_categorical_input_idx = [1,2,3,4]

        # network params
        self.batch_size = config['batch_size']
        self.valid_quantiles = config['vailid_quantiles']
        self.hidden_size = config['lstm_hidden_dimension']
        self.lstm_layers = config['lstm_layers']
        self.embedding_dim = config['embedding_dim']
        self.dropout = config['dropout']
        self.attn_heads = config['attn_heads']
        self.num_quantiles = config['num_quantiles']


        self.encode_length = config['encode_length']
        self.num_input_series_to_mask = config['num_masked_series']
        self.seq_length = config['seq_length']
        ## init embddings
        ### cat emb
        self.cat_embeddings = nn.LayerList()
        for i in range(self.time_varying_categoical_variables):
            emb = Embedding(config['time_varying_embedding_vocab_sizes'][i], config['embedding_dim'])
            self.cat_embeddings.append(emb)
        ### sta emb
        self.static_embedding_layers = nn.LayerList()
        for i in range(self.static_variables):
            emb = Embedding(config['static_embedding_vocab_sizes'][i], config['embedding_dim'])
            self.static_embedding_layers.append(emb)
        ### real emb
        self.time_varying_linear_layers = nn.LayerList()
        for i in range(self.time_varying_real_variables_encoder):
            emb = TimeDistributed(nn.Linear(1, config['embedding_dim']), batch_first=True)
            self.time_varying_linear_layers.append(emb)
        ### static variable select
        self.static_select = StaticVariableSelectionNetwork(self.embedding_dim, self.static_variables, self.hidden_size, self.dropout)
        self.static_grn1 = GRN(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout, False)
        self.static_grn2 = GRN(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout, False)
        self.static_grn3 = GRN(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout, False)
        self.static_grn4 = GRN(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout, False)
        ### hisotry and feature select
        # TODO: ctx
        self.hsitory_select = TimeVariableSelectionNetwork(self.encode_length, self.embedding_dim, 5, self.hidden_size, self.dropout, self.static_variables*self.hidden_size)
        self.feature_sleect = TimeVariableSelectionNetwork(self.seq_length-self.encode_length, self.embedding_dim, 4, self.hidden_size, self.dropout,self.static_variables*self.hidden_size)
        ### hsitory and feature lstm
        self.hsitory_lstm = nn.LSTM(self.hidden_size, self.hidden_size,time_major=False)
        self.feature_lstm = nn.LSTM(self.hidden_size, self.hidden_size,time_major=False)
        ### lstm glu add_and_norm
        self.lstm_glu = GLU(self.hidden_size, self.hidden_size, self.dropout)
        self.lstm_add_norm = Add_Norm(self.hidden_size)
        ### static encrichemtn 
        # TODO: ctx
        self.static_enh_grn = GRN(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout, True, add_ctx=self.static_variables*self.hidden_size)
        ### atten
        self.attn_layer = nn.MultiHeadAttention(self.hidden_size, self.attn_heads, self.dropout)
        self.attn_glu = GLU(self.hidden_size, self.hidden_size)
        self.attn_add_norm = Add_Norm(self.hidden_size)
        ## position wise feed-forward
        self.posi_ff_grn = GRN(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout, True)
        ### output
        self.out_glu = GLU(self.hidden_size, self.hidden_size)
        self.out_add_norm = Add_Norm(self.hidden_size)
        self.out_layer = Linear(self.hidden_size, self.num_quantiles, use_td=True)

    def forward(self, x):
        inputs = x['inputs'] # b t l
        num_reg = inputs.shape[-1] - self.time_varying_categoical_variables # 标量变量数
        num_cat = self.time_varying_categoical_variables # 类型变量数
        num_sta = self.static_variables # 统计变量数
        reg_inps = inputs[:,:,:num_reg] # b t l
        cat_inps = inputs[:,:,num_reg:] # b t l
        ## embedding inputs
        ### cat emb
        emb_cat_inps = [
            self.cat_embeddings[i](cat_inps[:,:,i])
            for i in range(num_cat)
        ]
        ### sta emb
        sta_inps = []
        # TODO: static 
        for i in range(self.static_variables):
            # only need static variable from the first timestep
            emb = self.static_embedding_layers[i](x['identifier'][:, 0, i])
            sta_inps.append(emb)
        sta_inps = paddle.stack(sta_inps, axis=1) # b l d
        if len(sta_inps.shape) == 2:
            sta_inps.unsqueeze_(1)
        # for i in range(self.static_variables):
        #     # only need static variable from the first timestep
        #     emb = self.static_embedding_layers[i](x['identifier'][:, 0, i])
        #     embedding_vectors.append(emb)
        # sta_inps = paddle.concat(embedding_vectors, axis=1)
        obs_inps = [
            self.time_varying_linear_layers[i](reg_inps[:,:,i:i+1])
            for i in range(num_reg)
        ]
        obs_inps = paddle.stack(obs_inps, axis=-1) # B T D L
        ### split unkown inp, kown inp
        # TODO: 处理
        unknow_inps = obs_inps[:,:,:,:1]
        know_reg_inps = obs_inps[:,:,:,1:]
        know_cat_inps = emb_cat_inps[0].unsqueeze(-1)
        know_inps = paddle.concat([know_reg_inps, know_cat_inps], axis=-1)

        if unknow_inps is None:
            history_inps = paddle.concat([
                know_inps[:,:self.encode_length, :],
                # obs_inps[:,:self.encode_length,:]
                ], axis=-1)
        else:
            history_inps = paddle.concat([
                unknow_inps[:, :self.encode_length, :],
                know_inps[:,:self.encode_length, :],
                # obs_inps[:,:self.encode_length,:]
                ], axis=-1)
        
        future_inps = know_inps[:, self.encode_length:, :]

        ## static variable selection
        static_encoder, static_weights = self.static_select(sta_inps)
        ## static encoder
        static_context_variable_selection = self.static_grn1(static_encoder)
        static_context_enrichment = self.static_grn2(static_encoder)
        static_context_state_h = self.static_grn3(static_encoder)
        static_context_state_c = self.static_grn4(static_encoder)

        ## variable selection
        historical_features, historical_flags = self.hsitory_select(
        history_inps, static_context_variable_selection) # B T Hide
        future_features, future_flags = self.feature_sleect(future_inps, static_context_variable_selection) # B T Hide

        # LSTM layer
        ## lstm encoder
        history_lstm, (state_h, state_c) \
        = self.hsitory_lstm(historical_features,
                                      initial_states=[static_context_state_h.unsqueeze(0),
                                                     static_context_state_c.unsqueeze(0)])
        ## lstm decoder
        future_lstm, _ = self.feature_lstm(
        future_features, initial_states=[state_h, state_c])
        ## apply gated skip connection
        lstm_layer = paddle.concat([history_lstm, future_lstm], axis=1) # B T H
        lstm_layer = self.lstm_glu(lstm_layer)
        lstm_layer = paddle.concat([history_lstm, future_lstm], axis=1)
        input_embeddings = paddle.concat([historical_features, future_features], axis=1)
        temporal_feature_layer = self.lstm_add_norm(lstm_layer, input_embeddings)

        # temporal fusion decoder
        ## static encrichemtn layers
        expanded_static_context = static_context_enrichment.unsqueeze(axis=1)
        enriched = self.static_enh_grn(temporal_feature_layer,expanded_static_context)
        ## temporal sefl-attention
        ### decoder self attention
        # mask = get_decoder_mask(enriched)
        mask = paddle.cumsum(paddle.eye(enriched.shape[1]), 1)
        x = self.attn_layer(enriched, enriched, enriched,
                          attn_mask=mask)
        x = self.attn_glu(x)
        x = self.attn_add_norm(x, enriched)
        ## position wise feed-forward
        decoder = self.posi_ff_grn(x)

        # output
        decoder = self.out_glu(decoder)
        transformer_layer = self.out_add_norm(decoder, temporal_feature_layer)
        outputs = self.out_layer(transformer_layer[:,self.encode_length:,:])

        return outputs
