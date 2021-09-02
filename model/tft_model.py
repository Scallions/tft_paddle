# from model.model import Embedding
import paddle
import paddle.nn as nn
import json

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

class Embedding(nn.Layer):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, sparse=False, weight_attr=None, name=None):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx, sparse, weight_attr, name)
    
    def forward(self, x):
        return self.emb(x.astype('int64'))

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


class ScaledDotProductAttention(nn.Layer):
    """Defines scaled dot product attention layer.
        Attributes:
        dropout: Dropout rate to use
        activation: Normalisation function for scaled dot product attention (e.g.
            softmax by default)
    """

    def __init__(self, attn_dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.activation = nn.Softmax()

    def forward(self, q, k, v, mask):
        """Applies scaled dot product attention.
        Args:
            q: Queries
            k: Keys
            v: Values
            mask: Masking if required -- sets softmax to very large value
        Returns:
            Tuple of (layer outputs, attention weights)
        """
        attn = paddle.bmm(q,k.transpose([0,2,1])) # shape=(batch, q, k)
        temper = k.shape[-1]**0.5
        attn = attn / temper
        if mask is not None:
            # attn = attn.masked_fill(mask.bool(), -1e9)
            ## 1
            # fill = -1e9 * mask
            # fmask = -1 * (mask - 1)
            # attn = fmask * attn + fill
            ## attn = paddle.masked_selet(attn, mask.bool())
            ## google
            # REVIEW
            mask = -1e9 * (1-mask)
            attn = attn + mask

        attn = self.activation(attn)
        attn = self.dropout(attn)
        output = paddle.bmm(attn,v)
        return output, attn

class InterpretableMultiHeadAttention(nn.Layer):
    """Defines interpretable multi-head attention layer.
    Attributes:
      n_head: Number of heads
      d_k: Key/query dimensionality per head
      d_v: Value dimensionality
      dropout: Dropout rate to apply
      qs_layers: List of queries across heads
      ks_layers: List of keys across heads
      vs_layers: List of values across heads
      attention: Scaled dot product attention layer
      w_o: Output weight matrix to project internal state to the original TFT
        state size
    """

    def __init__(self, n_head, d_model, dropout_rate):
        """Initialises layer.
        Args:
          n_head: Number of heads
          d_model: TFT state dimensionality
          dropout: Dropout discard rate
        """
        super().__init__()

        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = nn.Dropout(dropout_rate)

        self.qs_layers = nn.LayerList()
        self.ks_layers = nn.LayerList()
        self.vs_layers = nn.LayerList()

        # Use same value layer to facilitate interp
        vs_layer = nn.Linear(d_model, d_k, bias_attr=False)
        # REVIEW

        for _ in range(n_head):
            self.qs_layers.append(Linear(d_model, d_v, bias=False))
            self.ks_layers.append(Linear(d_model, d_v, bias=False))
            self.vs_layers.append(vs_layer)  # use same vs_layer

        self.attention = ScaledDotProductAttention()
        self.w_o = Linear(self.d_k, d_model, bias=False)

    def forward(self, q, k, v, attn_mask=None):
        """Applies interpretable multihead attention.
          Using T to denote the number of time steps fed into the transformer.
          Args:
            q: Query tensor of shape=(?, T, d_model)
            k: Key of shape=(?, T, d_model)
            v: Values of shape=(?, T, d_model)
            mask: Masking if required with shape=(?, T, T)
          Returns:
            Tuple of (layer outputs, attention weights)
          """
        n_head = self.n_head
        heads = []
        attns = []
        for i in range(n_head):
            qs = self.qs_layers[i](q)
            ks = self.ks_layers[i](k)
            vs = self.vs_layers[i](v)
            head, attn = self.attention(qs, ks, vs, attn_mask)

            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)
        head = paddle.stack(heads) if n_head > 1 else heads[0]
        attn = paddle.stack(attns)

        outputs = paddle.mean(head, axis=0) if n_head > 1 else head
        outputs = self.w_o(outputs)
        outputs = self.dropout(outputs)  # output dropout

        # return outputs, attn
        return outputs

class Linear(nn.Layer):
    def __init__(self, inp_size, out_size, use_td=False, bias=True):
        super().__init__()
        weight_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.XavierUniform())
        if not bias:
            self.linear = nn.Linear(inp_size, out_size, bias_attr=False, weight_attr=weight_attr)
        else:
            self.linear = nn.Linear(inp_size, out_size, weight_attr=weight_attr)
        if use_td:
            self.linear = TimeDistributed(self.linear)

    def forward(self, x):
        return self.linear(x)

class GLU(nn.Layer):
    # Gated Linear Unit
    ## REVIEW
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
        fc1 = self.fc1(x)
        sig = self.sigmoid(self.fc2(x))
        return paddle.multiply(sig, fc1)

class Add_Norm(nn.Layer):
    def __init__(self, size):
        super().__init__()
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
            self.skip_layer = Linear(self.input_size, self.output_size, use_td)
        
        ## feedforward network
        self.hidden_layer = Linear(self.input_size, self.hidden_state_size, use_td)
        
        self.add_ctx = None
        if add_ctx is not None:
            # self.add_ctx = Linear(hidden_state_size, hidden_state_size, use_td)
            self.add_ctx = Linear(add_ctx, hidden_state_size, use_td)

        self.elu = nn.ELU()
        self.hidden_layer2 = Linear(hidden_state_size, hidden_state_size, use_td)
        self.gating_layer = GLU(hidden_state_size, output_size, dropout, use_td)
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
        flatten = self.flatten(embedding) # b 160
        if context is not None:
            mlp_outputs = self.flattened_grn(flatten, context)
        else:
            mlp_outputs = self.flattened_grn(flatten)
        sparse_weights = self.softmax(mlp_outputs).unsqueeze(2)

        trans_emb_list = []
        for i in range(self.num_inputs):
            # trans_emb_list.append(
            #     self.single_variable_grns[i](embedding[:, :, (i * self.input_size): (i + 1) * self.input_size]))
            ## REVIEW
            trans_emb_list.append(
                self.single_variable_grns[i](embedding[:, i: i + 1, :]))
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
                                                      self.hidden_size, self.dropout,True, self.context)
        else:
            self.flattened_grn = GRN(self.embedding_dim * self.input_nums, self.input_nums,
                                                      self.hidden_size, self.dropout, True)
        self.single_variable_grns = nn.LayerList()
        for i in range(self.input_nums):
            self.single_variable_grns.append(
                GRN(self.embedding_dim, self.hidden_size, self.hidden_size, self.dropout, True))
        self.softmax = nn.Softmax()

    def forward(self, embedding, context=None):
        # flatten = self.flatten(embedding)
        flatten = embedding.reshape([-1, self.time_steps, self.embedding_dim * self.input_nums])
        context = context.unsqueeze(1)
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
    def __init__(self, raw_params):
        super().__init__()
        params = dict(raw_params)  # copy locally
        print(params)

        ## data params
        self.input_size = int(params['input_size'])
        self.output_size = int(params['output_size'])
        self.input_obs_loc = json.loads(str(params['input_obs_loc']))
        self.static_loc = json.loads(str(params['static_input_loc']))
        self.static_variables = len(self.static_loc)
        self.cat_counts = json.loads(str(params['category_counts']))
        self.know_reg = json.loads(
            str(params['known_regular_inputs']))
        self.know_cat = json.loads(
            str(params['known_categorical_inputs']))

        # network params
        self.batch_size = int(params['batch_size'])
        self.hidden_size = int(params['hidden_layer_size'])
        self.dropout = float(params['dropout_rate'])
        self.attn_heads = int(params['num_heads'])
        self.num_quantiles = len(list(params['quantiles']))


        self.encode_length = int(params['num_encoder_steps'])
        self.seq_length = int(params['total_time_steps'])
        ## init embddings
        ### cat emb
        self.cat_embeddings = nn.LayerList()
        for i in range(len(self.cat_counts)):
            emb = Embedding(self.cat_counts[i], self.hidden_size)
            # emb = nn.Embedding(self.cat_counts[i], self.hidden_size)
            self.cat_embeddings.append(emb)
        ### sta emb
        self.static_embedding_layers = nn.LayerList()
        for i in range(self.static_variables):
            emb = Linear(1, self.hidden_size)
            self.static_embedding_layers.append(emb)
        ### real emb
        self.reg_embedding_layers = nn.LayerList()
        for i in range(self.input_size - len(self.cat_counts)):
            # emb = TimeDistributed(nn.Linear(1, self.hidden_size))
            emb = Linear(1, self.hidden_size, use_td=True)
            self.reg_embedding_layers.append(emb)
        ### static variable select
        self.static_select = StaticVariableSelectionNetwork(self.hidden_size, self.static_variables, self.hidden_size, self.dropout)
        self.static_grn1 = GRN(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout, False)
        self.static_grn2 = GRN(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout, False)
        self.static_grn3 = GRN(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout, False)
        self.static_grn4 = GRN(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout, False)
        ### hisotry and feature select
        self.history_select = TimeVariableSelectionNetwork(self.encode_length, self.hidden_size, 4, self.hidden_size, self.dropout, self.static_variables*self.hidden_size)
        self.future_select = TimeVariableSelectionNetwork(self.seq_length-self.encode_length, self.hidden_size, 3, self.hidden_size, self.dropout,self.static_variables*self.hidden_size)
        ### history and feature lstm
        from scipy.stats import ortho_group
        dim = (self.input_size-self.output_size) * self.hidden_size
        his_whh = ortho_group.rvs(size=2, dim=dim)[0][:,:self.hidden_size]
        fut_whh = ortho_group.rvs(size=2, dim=dim)[0][:,:self.hidden_size]
        self.history_lstm = nn.LSTM(self.hidden_size, self.hidden_size,time_major=False,
            weight_ih_attr = paddle.framework.ParamAttr(name="history_weight_ih",initializer=paddle.nn.initializer.XavierUniform()),
            weight_hh_attr = paddle.framework.ParamAttr(name="history_weight_hh",initializer=paddle.nn.initializer.Assign(his_whh)),
            )
        self.future_lstm = nn.LSTM(self.hidden_size, self.hidden_size,time_major=False,
            weight_ih_attr = paddle.framework.ParamAttr(name="future_weight_ih",initializer=paddle.nn.initializer.XavierUniform()),
            weight_hh_attr = paddle.framework.ParamAttr(name="future_weight_hh",initializer=paddle.nn.initializer.Assign(fut_whh)),
            )
        ### lstm glu add_and_norm
        self.lstm_glu = GLU(self.hidden_size, self.hidden_size, self.dropout)
        self.lstm_add_norm = Add_Norm(self.hidden_size)
        ### static encrichemtn 
        self.static_enh_grn = GRN(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout, True, add_ctx=self.static_variables*self.hidden_size)
        ### atten
        self.attn_layer = InterpretableMultiHeadAttention(self.attn_heads, self.hidden_size, dropout_rate=self.dropout)
        # self.attn_layer = nn.MultiHeadAttention(self.hidden_size, self.attn_heads, self.dropout)
        self.attn_glu = GLU(self.hidden_size, self.hidden_size)
        self.attn_add_norm = Add_Norm(self.hidden_size)
        ## position wise feed-forward
        self.posi_ff_grn = GRN(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout, True)
        ### output
        self.out_glu = GLU(self.hidden_size, self.hidden_size)
        self.out_add_norm = Add_Norm(self.hidden_size)
        self.out_layer = Linear(self.hidden_size, self.num_quantiles, use_td=True)

    def forward(self, x):
        # inputs = x['inputs'] # b t l b 192 5
        inputs = x
        num_cat = len(self.cat_counts) # 类型变量数 1 
        num_reg = self.input_size - num_cat # 标量变量数 4
        num_sta = self.static_variables # 统计变量数 1
        reg_inps = inputs[:,:,:num_reg] # b t l   b 192 4
        cat_inps = inputs[:,:,num_reg:] # b t l   b 192 1
        ## embedding inputs
        ### cat emb
        emb_cat_inps = [
            self.cat_embeddings[i](cat_inps[:,:,i]) # b 192 160   params_count 59040
            for i in range(num_cat)
        ]
        ### sta emb
        sta_inps = [ self.static_embedding_layers[i](x['identifier'][:, 0, i]) for i in range(num_reg) if i in self.static_loc] \
             + [ emb_cat_inps[i][:,0,:] for i in range(num_cat) if i+num_reg in self.static_loc] # b 160
        sta_inps = paddle.stack(sta_inps, axis=1) # b l d  b 1 160
        ### real emb
        obs_inps = [
            self.reg_embedding_layers[i](reg_inps[:,:,i:i+1])
            for i in self.input_obs_loc
        ]
        obs_inps = paddle.stack(obs_inps, axis=-1) # B T D L
        ### split unkown inp, kown inp
        wired_embeddings = []
        for i in range(num_cat):
            if i not in self.know_cat \
                and  i + num_reg  not in self.input_obs_loc:
                # e = self.cat_embeddings[i](cat_inps[:, :, i])
                e = emb_cat_inps[:,:,i]
                wired_embeddings.append(e)

        unknow_inps = []
        for i in range(reg_inps.shape[-1]):
            if i not in self.know_reg \
                and i not in self.input_obs_loc:
                e = self.reg_embedding_layers[i](reg_inps[:,:, i:i + 1])
                unknow_inps.append(e)
        if unknow_inps + wired_embeddings:
            unknow_inps = paddle.stack(
                unknow_inps + wired_embeddings, axis=-1)
        else:
            unknow_inps = None
        # A priori known inputs
        known_regular_inputs = [
            self.reg_embedding_layers[i](reg_inps[:,:, i:i + 1])
            for i in self.know_reg
            if i not in self.static_loc
        ]
        known_categorical_inputs = [
            emb_cat_inps[i]
            for i in self.know_cat
            if i + num_reg not in self.static_loc
        ]
        know_inps = paddle.stack(known_regular_inputs + known_categorical_inputs, axis=-1)


        if unknow_inps is None:
            history_inps = paddle.concat([
                know_inps[:,:self.encode_length, :],
                obs_inps[:,:self.encode_length,:]
                ], axis=-1)
        else:
            history_inps = paddle.concat([
                unknow_inps[:, :self.encode_length, :],
                know_inps[:,:self.encode_length, :],
                obs_inps[:,:self.encode_length,:]
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
        historical_features, historical_flags = self.history_select(
        history_inps, static_context_variable_selection) # B T Hide
        future_features, future_flags = self.future_select(future_inps, static_context_variable_selection) # B T Hide

        # LSTM layer
        ## lstm encoder
        history_lstm, (state_h, state_c) \
        = self.history_lstm(historical_features,
                                      initial_states=[static_context_state_h.unsqueeze(0),
                                                     static_context_state_c.unsqueeze(0)])
        ## lstm decoder
        future_lstm, _ = self.future_lstm(
        future_features, initial_states=[state_h, state_c])
        ## apply gated skip connection
        lstm_layer = paddle.concat([history_lstm, future_lstm], axis=1) # B T H
        lstm_layer = self.lstm_glu(lstm_layer)
        input_embeddings = paddle.concat([historical_features, future_features], axis=1)
        temporal_feature_layer = self.lstm_add_norm(lstm_layer, input_embeddings)

        # temporal fusion decoder
        ## static encrichemtn layers
        expanded_static_context = static_context_enrichment.unsqueeze(axis=1)
        enriched = self.static_enh_grn(temporal_feature_layer,expanded_static_context)
        ## temporal sefl-attention
        ### decoder self attention
        # mask = get_decoder_mask(enriched)
        mask = paddle.cumsum(paddle.ones((enriched.shape[0],1,1))*paddle.eye(enriched.shape[1]), 1)
        # mask = paddle.cumsum(paddle.eye(enriched.shape[1]).reshape((1, enriched.shape[1], enriched.shape[1])).repeat(enriched.shape[0], 1, 1), 1)
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
