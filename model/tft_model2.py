import paddle
import paddle.nn as nn

class TimeDistributed(nn.Layer):
    ## Takes any module and stacks the time dimension with the batch dimenison of inputs before apply the module
    ## From: https://discuss.pypaddle.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.shape) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.reshape([-1, x.shape[-1]])  # (samples * timesteps, input_size)

        if x_reshape.dtype != paddle.float32:
            x_reshape = x_reshape.astype('float32')

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.reshape([x.shape[0], -1, y.shape[-1]])  # (samples, timesteps, output_size)
        else:
            y = y.reshape([-1, x.shape[1], y.shape[-1]])  # (timesteps, samples, output_size)

        return y

class LinearLayer(nn.Layer):
    def __init__(self,
                input_size,
                size,
                use_time_distributed=True,
                batch_first=False):
        super(LinearLayer, self).__init__()

        self.use_time_distributed=use_time_distributed
        self.input_size=input_size
        self.size=size
        if use_time_distributed:
            self.layer = TimeDistributed(nn.Linear(input_size, size), batch_first=batch_first)
        else:
            self.layer = nn.Linear(input_size, size)
      
    def forward(self, x):
        return self.layer(x)

class AddAndNorm(nn.Layer):
    def __init__(self, hidden_layer_size):
        super(AddAndNorm, self).__init__()

        self.normalize = nn.LayerNorm(hidden_layer_size)

    def forward(self, x1, x2):
        x = paddle.add(x1, x2)
        return self.normalize(x)


class GLU(nn.Layer):
    #Gated Linear Unit
    def __init__(self, 
                input_size,
                hidden_layer_size,
                dropout_rate=None,
                use_time_distributed=True,
                batch_first=False
                ):
        super(GLU, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        self.use_time_distributed = use_time_distributed

        if dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)
        
        self.activation_layer = LinearLayer(input_size, hidden_layer_size, use_time_distributed, batch_first)
        self.gated_layer = LinearLayer(input_size, hidden_layer_size, use_time_distributed, batch_first)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        if self.dropout_rate is not None:
            x = self.dropout(x)
        
        activation = self.activation_layer(x)
        gated = self.sigmoid(self.gated_layer(x))
        
        return paddle.multiply(activation, gated), gated


class GatedResidualNetwork(nn.Layer):
    def __init__(self, 
                input_size,
                hidden_layer_size,
                output_size=None,
                dropout_rate=None,
                use_time_distributed=True,
                return_gate=False,
                batch_first=False
                ):

        super(GatedResidualNetwork, self).__init__()
        if output_size is None:
            output = hidden_layer_size
        else:
            output = output_size
        
        self.output = output
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.return_gate = return_gate

        self.linear_layer = LinearLayer(input_size, output, use_time_distributed, batch_first)

        self.hidden_linear_layer1 = LinearLayer(input_size, hidden_layer_size, use_time_distributed, batch_first)
        self.hidden_context_layer = LinearLayer(hidden_layer_size, hidden_layer_size, use_time_distributed, batch_first)
        self.hidden_linear_layer2 = LinearLayer(hidden_layer_size, hidden_layer_size, use_time_distributed, batch_first)

        self.elu1 = nn.ELU()
        self.glu = GLU(hidden_layer_size, output, dropout_rate, use_time_distributed, batch_first)
        self.add_and_norm = AddAndNorm(hidden_layer_size=output)

    def forward(self, x, context=None):
        # Setup skip connection
        if self.output_size is None:
            skip = x
        else:
            skip = self.linear_layer(x)

        # Apply feedforward network
        hidden = self.hidden_linear_layer1(x)
        if context is not None:
            hidden = hidden + self.hidden_context_layer(context)
        hidden = self.elu1(hidden)
        hidden = self.hidden_linear_layer2(hidden)

        gating_layer, gate = self.glu(hidden)
        if self.return_gate:
            return self.add_and_norm(skip, gating_layer), gate
        else:
            return self.add_and_norm(skip, gating_layer)

class StaticCombineAndMask(nn.Layer):
    def __init__(self, input_size, num_static, hidden_layer_size, dropout_rate, additional_context=None, use_time_distributed=False, batch_first=True):
        super(StaticCombineAndMask, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.input_size =input_size
        self.num_static = num_static
        self.dropout_rate = dropout_rate
        self.additional_context = additional_context

        if self.additional_context is not None:
            self.flattened_grn = GatedResidualNetwork(self.num_static*self.hidden_layer_size, self.hidden_layer_size, self.num_static, self.dropout_rate, use_time_distributed=False, return_gate=False, batch_first=batch_first)
        else:
            self.flattened_grn = GatedResidualNetwork(self.num_static*self.hidden_layer_size, self.hidden_layer_size, self.num_static, self.dropout_rate, use_time_distributed=False, return_gate=False, batch_first=batch_first)


        self.single_variable_grns = nn.LayerList()
        for i in range(self.num_static):
            self.single_variable_grns.append(GatedResidualNetwork(self.hidden_layer_size, self.hidden_layer_size, None, self.dropout_rate, use_time_distributed=False, return_gate=False, batch_first=batch_first))

<<<<<<< HEAD
        self.softmax = nn.Softmax()
=======
        self.softmax = nn.Softmax(axis=1)
>>>>>>> develop

    def forward(self, embedding, additional_context=None):
        # Add temporal features
        _, num_static, _ = list(embedding.shape)
        flattened_embedding = paddle.flatten(embedding, start_axis=1)
        if additional_context is not None:
            sparse_weights = self.flattened_grn(flattened_embedding, additional_context)
        else:
            sparse_weights = self.flattened_grn(flattened_embedding)

        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)

        trans_emb_list = []
        for i in range(self.num_static):
            ##select slice of embedding belonging to a single input
            trans_emb_list.append(
              self.single_variable_grns[i](paddle.flatten(embedding[:, i:i + 1, :], start_axis=1))
            )

        transformed_embedding = paddle.stack(trans_emb_list, axis=1)

        combined = transformed_embedding*sparse_weights
        
        static_vec = combined.sum(axis=1)

        return static_vec, sparse_weights

class LSTMCombineAndMask(nn.Layer):
    def __init__(self, input_size, num_inputs, hidden_layer_size, dropout_rate, use_time_distributed=False, batch_first=True):
        super(LSTMCombineAndMask, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.dropout_rate = dropout_rate
        
        self.flattened_grn = GatedResidualNetwork(self.num_inputs*self.hidden_layer_size, self.hidden_layer_size, self.num_inputs, self.dropout_rate, use_time_distributed=use_time_distributed, return_gate=True, batch_first=batch_first)

        self.single_variable_grns = nn.LayerList()
        for i in range(self.num_inputs):
            self.single_variable_grns.append(GatedResidualNetwork(self.hidden_layer_size, self.hidden_layer_size, None, self.dropout_rate, use_time_distributed=use_time_distributed, return_gate=False, batch_first=batch_first))

        # TODO: change dim
<<<<<<< HEAD
        self.softmax = nn.Softmax()
=======
        self.softmax = nn.Softmax(axis=2)
>>>>>>> develop

    def forward(self, embedding, additional_context=None):
        # Add temporal features
        _, time_steps, embedding_dim, num_inputs = list(embedding.shape)
                
        flattened_embedding = paddle.reshape(embedding,
                      [-1, time_steps, embedding_dim * num_inputs])

        expanded_static_context = additional_context.unsqueeze(1)

        if additional_context is not None:
            sparse_weights, static_gate = self.flattened_grn(flattened_embedding, expanded_static_context)
        else:
            sparse_weights = self.flattened_grn(flattened_embedding)

        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)

        trans_emb_list = []
        for i in range(self.num_inputs):
            ##select slice of embedding belonging to a single input
            trans_emb_list.append(
              self.single_variable_grns[i](embedding[:,:,:,i])
            )

        transformed_embedding = paddle.stack(trans_emb_list, axis=-1)
        
        combined = transformed_embedding*sparse_weights
        
        temporal_ctx = combined.sum(axis=-1)

        return temporal_ctx, sparse_weights, static_gate

class ScaledDotProductAttention(nn.Layer):
    """Defines scaled dot product attention layer.

      Attributes:
        dropout: Dropout rate to use
        activation: Normalisation function for scaled dot product attention (e.g.
          softmax by default)
    """

    def __init__(self, attn_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()

        self.dropout = nn.Dropout(attn_dropout)
<<<<<<< HEAD
        self.activation = nn.Softmax()
=======
        self.activation = nn.Softmax(axis=-1)
>>>>>>> develop
        # self.device = paddle.device('cuda' if paddle.cuda.is_available() else 'cpu')

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
        """ TODO: change
        attn = paddle.bmm(q,k.permute(0,2,1)) # shape=(batch, q, k)
        if mask is not None:
            attn = attn.masked_fill(mask.bool().to(self.device), -1e9)

        attn = self.activation(attn)
        attn = self.dropout(attn)
        output = paddle.bmm(attn,v)
        return output, attn
        """
        attn = paddle.bmm(q,k.transpose([0,2,1])) # shape=(batch, q, k)
        if mask is not None:
            # attn = attn.masked_fill(mask.bool(), -1e9)
            fill = -1e9 * mask
            fmask = -1 * (mask - 1)
            attn = fmask * attn + fill
            # attn = paddle.masked_selet(attn, mask.bool())

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
        super(InterpretableMultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = nn.Dropout(dropout_rate)

        self.qs_layers = nn.LayerList()
        self.ks_layers = nn.LayerList()
        self.vs_layers = nn.LayerList()

        # Use same value layer to facilitate interp
        vs_layer = nn.Linear(d_model, d_v, bias_attr=False)
        qs_layer = nn.Linear(d_model, d_k, bias_attr=False)
        ks_layer = nn.Linear(d_model, d_k, bias_attr=False)
        # TODO: check same layer
        for _ in range(n_head):
            self.qs_layers.append(qs_layer)
            self.ks_layers.append(ks_layer)
            self.vs_layers.append(vs_layer)  # use same vs_layer

        self.attention = ScaledDotProductAttention()
        self.w_o = nn.Linear(self.d_k, d_model, bias_attr=False)

    def forward(self, q, k, v, mask=None):
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
            head, attn = self.attention(qs, ks, vs, mask)

            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attns.append(attn)
        head = paddle.stack(heads) if n_head > 1 else heads[0]
        attn = paddle.stack(attns)

        outputs = paddle.mean(head, axis=0) if n_head > 1 else head
        outputs = self.w_o(outputs)
        outputs = self.dropout(outputs)  # output dropout

        return outputs, attn


class TFT(nn.Layer):
    def __init__(self, config):
        super(TFT, self).__init__()

        # params = dict(raw_params)  # copy locally
        print(config)

        # Data parameters
        # self.time_steps = int(config['total_time_steps'])
        self.time_steps = 192
        # self.input_size = int(config['input_size'])
        self.input_size = 5
        # self.output_size = int(config['output_size'])
        self.output_size = 1
        # self.category_counts = json.loads(str(config['category_counts']))
        self.category_counts = [369]
        # self.n_multiprocessing_workers = int(config['n_workers'])

        # Relevant indices for TFT
        # self._input_obs_loc = json.loads(str(config['input_obs_loc']))
        self._input_obs_loc = [0]
        # self._static_input_loc = json.loads(str(config['static_input_loc']))
        self._static_input_loc = [4]
        # self._known_regular_inputidx = json.loads(
            # str(config['known_regular_inputs']))
        self._known_regular_input_idx = [1,2,3]
        # self._known_categorical_input_idx = json.loads(
            # str(config['known_categorical_inputs']))
        self._known_categorical_input_idx = [0]
        # Network params
        # self.quantiles = list(config['quantiles'])
        self.quantiles = [0.1,0.5,0.9]
        # self.device = str(config['device'])
        # self.hidden_layer_size = int(config['hidden_layer_size'])
        self.hidden_layer_size = 160
        # self.dropout_rate = float(config['dropout_rate'])
        self.dropout_rate = 0.1
        # self.max_gradient_norm = float(config['max_gradient_norm'])
        # self.learning_rate = float(config['lr'])
        # self.minibatch_size = int(config['batch_size'])
        # self.num_epochs = int(config['num_epochs'])
        # self.early_stopping_patience = int(config['early_stopping_patience'])

        # self.num_encoder_steps = int(config['num_encoder_steps'])
        self.num_encoder_steps = 168
        # self.num_stacks = int(config['stack_size'])
        self.num_stacks = 1
        # self.num_heads = int(config['num_heads'])
        self.num_heads = 4
        self.batch_first = True
        self.num_static = len(self._static_input_loc)
        self.num_inputs = len(self._known_regular_input_idx) + self.output_size
        self.num_inputs_decoder = len(self._known_regular_input_idx)

        # Serialisation options
        # self._temp_folder = os.path.join(config['model_folder'], 'tmp')
        # self.reset_temp_folder()

        # Extra components to store Tensorflow nodes for attention computations
        self._input_placeholder = None
        self._attention_components = None
        self._prediction_parts = None

        # print('*** params ***')
        # for k in params:
        #   print('# {} = {}'.format(k, config[k]))

        #######
        time_steps = self.time_steps
        num_categorical_variables = len(self.category_counts)
        num_regular_variables = self.input_size - num_categorical_variables

        embedding_sizes = [
          self.hidden_layer_size for i, size in enumerate(self.category_counts)
        ]

        print("num_categorical_variables")
        print(num_categorical_variables)
        self.embeddings = nn.LayerList()
        for i in range(num_categorical_variables):
            embedding = nn.Embedding(self.category_counts[i], embedding_sizes[i])
            self.embeddings.append(embedding)

        self.static_input_layer = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.time_varying_embedding_layer = LinearLayer(input_size=1, size=self.hidden_layer_size, use_time_distributed=True, batch_first=self.batch_first)

        self.static_combine_and_mask = StaticCombineAndMask(
                input_size=self.input_size,
                num_static=self.num_static,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                additional_context=None,
                use_time_distributed=False,
                batch_first=self.batch_first)
        self.static_context_variable_selection_grn = GatedResidualNetwork(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                return_gate=False,
                batch_first=self.batch_first)
        self.static_context_enrichment_grn = GatedResidualNetwork(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                return_gate=False,
                batch_first=self.batch_first)
        self.static_context_state_h_grn = GatedResidualNetwork(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                return_gate=False,
                batch_first=self.batch_first)
        self.static_context_state_c_grn = GatedResidualNetwork(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                return_gate=False,
                batch_first=self.batch_first)
        self.historical_lstm_combine_and_mask = LSTMCombineAndMask(
                input_size=self.num_encoder_steps,
                num_inputs=self.num_inputs,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                batch_first=self.batch_first)
        self.future_lstm_combine_and_mask = LSTMCombineAndMask(
                input_size=self.num_encoder_steps,
                num_inputs=self.num_inputs_decoder,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                batch_first=self.batch_first)
        # TODO: time major
        self.lstm_encoder = nn.LSTM(input_size=self.hidden_layer_size, hidden_size=self.hidden_layer_size, time_major=not self.batch_first)
        self.lstm_decoder = nn.LSTM(input_size=self.hidden_layer_size, hidden_size=self.hidden_layer_size, time_major=not self.batch_first)

        self.lstm_glu = GLU(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                batch_first=self.batch_first)
        self.lstm_glu_add_and_norm = AddAndNorm(hidden_layer_size=self.hidden_layer_size)

        self.static_enrichment_grn = GatedResidualNetwork(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                return_gate=True,
                batch_first=self.batch_first)

        self.self_attn_layer = InterpretableMultiHeadAttention(self.num_heads, self.hidden_layer_size, dropout_rate=self.dropout_rate)

        self.self_attention_glu = GLU(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                batch_first=self.batch_first)
        self.self_attention_glu_add_and_norm = AddAndNorm(hidden_layer_size=self.hidden_layer_size)

        self.decoder_grn = GatedResidualNetwork(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                output_size=None,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                return_gate=False,
                batch_first=self.batch_first)

        self.final_glu = GLU(
                input_size=self.hidden_layer_size,
                hidden_layer_size=self.hidden_layer_size,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                batch_first=self.batch_first)
        self.final_glu_add_and_norm = AddAndNorm(hidden_layer_size=self.hidden_layer_size)

        self.output_layer = LinearLayer(
                input_size=self.hidden_layer_size,
                size=self.output_size * len(self.quantiles),
                use_time_distributed=True,
                batch_first=self.batch_first)

    def get_decoder_mask(self, self_attn_inputs):
        """Returns causal mask to apply for self-attention layer.

        Args:
          self_attn_inputs: Inputs to self attention layer to determine mask shape
        """
        len_s = self_attn_inputs.shape[1] # 192
        bs = self_attn_inputs.shape[:1][0] # [64]
        # create batch_size identity matrices
        # TODO: check
        mask = paddle.cumsum(paddle.ones([bs,1,1]) * paddle.eye(len_s).reshape((1, len_s, len_s)), 1)
        return mask

    def get_tft_embeddings(self, all_inputs):
        time_steps = self.time_steps

        num_categorical_variables = len(self.category_counts)
        num_regular_variables = self.input_size - num_categorical_variables

        embedding_sizes = [
          self.hidden_layer_size for i, size in enumerate(self.category_counts)
        ]

        regular_inputs, categorical_inputs \
            = all_inputs[:, :, :num_regular_variables], \
              all_inputs[:, :, num_regular_variables:]

        embedded_inputs = [
            self.embeddings[i](categorical_inputs[:,:, i].astype('long'))
            for i in range(num_categorical_variables)
        ]

        # Static inputs
        if self._static_input_loc:
            static_inputs = []
            for i in range(num_regular_variables):
                if i in self._static_input_loc:
                    reg_i = self.static_input_layer(regular_inputs[:, 0, i:i + 1])
                    static_inputs.append(reg_i)

            emb_inputs = []
            for i in range(num_categorical_variables):
                if i + num_regular_variables in self._static_input_loc:
                    emb_inputs.append(embedded_inputs[i][:, 0, :])

            static_inputs += emb_inputs
            static_inputs = paddle.stack(static_inputs, axis=1)

        else:
            static_inputs = None

        # Targets
        obs_inputs = paddle.stack([
            self.time_varying_embedding_layer(regular_inputs[:,:, i:i + 1].astype('float'))
            for i in self._input_obs_loc
        ], axis=-1)


        # Observed (a prioir unknown) inputs
        wired_embeddings = []
        for i in range(num_categorical_variables):
            if i not in self._known_categorical_input_idx and i not in self._input_obs_loc:
                e = self.embeddings[i](categorical_inputs[:, :, i])
                wired_embeddings.append(e)

        unknown_inputs = []
        for i in range(regular_inputs.shape[-1]):
            if i not in self._known_regular_input_idx and i not in self._input_obs_loc:
                e = self.time_varying_embedding_layer(regular_inputs[:,:, i:i + 1])
                unknown_inputs.append(e)

        if unknown_inputs + wired_embeddings:
            unknown_inputs = paddle.stack(unknown_inputs + wired_embeddings, axis=-1)
        else:
            unknown_inputs = None

        # A priori known inputs
        known_regular_inputs = []
        for i in self._known_regular_input_idx:
            if i not in self._static_input_loc:
                known_regular_inputs.append(self.time_varying_embedding_layer(regular_inputs[:,:, i:i + 1].astype('float')))

        known_categorical_inputs = []
        for i in self._known_categorical_input_idx:
            if i + num_regular_variables not in self._static_input_loc:
                known_categorical_inputs.append(embedded_inputs[i])

        known_combined_layer = paddle.stack(known_regular_inputs + known_categorical_inputs, axis=-1)

        return unknown_inputs, known_combined_layer, obs_inputs, static_inputs

    def forward(self, x):
        # Size definitions.
        time_steps = self.time_steps
        combined_input_size = self.input_size
        encoder_steps = self.num_encoder_steps
        all_inputs = x['inputs']

        unknown_inputs, known_combined_layer, obs_inputs, static_inputs \
            = self.get_tft_embeddings(all_inputs)

        # Isolate known and observed historical inputs.
        if unknown_inputs is not None:
            historical_inputs = paddle.concat([
                unknown_inputs[:, :encoder_steps, :],
                known_combined_layer[:, :encoder_steps, :],
                obs_inputs[:, :encoder_steps, :]
            ], axis=-1)
        else:
            historical_inputs = paddle.concat([
                  known_combined_layer[:, :encoder_steps, :],
                  obs_inputs[:, :encoder_steps, :]
              ], axis=-1)

        # Isolate only known future inputs.
        future_inputs = known_combined_layer[:, encoder_steps:, :]

        static_encoder, static_weights = self.static_combine_and_mask(static_inputs)
        static_context_variable_selection = self.static_context_variable_selection_grn(static_encoder)
        static_context_enrichment = self.static_context_enrichment_grn(static_encoder)
        static_context_state_h = self.static_context_state_h_grn(static_encoder)
        static_context_state_c = self.static_context_state_c_grn(static_encoder)
        historical_features, historical_flags, _ = self.historical_lstm_combine_and_mask(historical_inputs, static_context_variable_selection)
        future_features, future_flags, _ = self.future_lstm_combine_and_mask(future_inputs, static_context_variable_selection)

        history_lstm, (state_h, state_c) = self.lstm_encoder(historical_features, (static_context_state_h.unsqueeze(0), static_context_state_c.unsqueeze(0)))
        future_lstm, _ = self.lstm_decoder(future_features, (state_h, state_c))

        lstm_layer = paddle.concat([history_lstm, future_lstm], axis=1)
        # Apply gated skip connection
        input_embeddings = paddle.concat([historical_features, future_features], axis=1)

        lstm_layer, _ = self.lstm_glu(lstm_layer)
        temporal_feature_layer = self.lstm_glu_add_and_norm(lstm_layer, input_embeddings)

        # Static enrichment layers
        expanded_static_context = static_context_enrichment.unsqueeze(1)
        enriched, _ = self.static_enrichment_grn(temporal_feature_layer, expanded_static_context)

        # Decoder self attention
        mask = self.get_decoder_mask(enriched)
        x, self_att = self.self_attn_layer(enriched, enriched, enriched, mask)#, attn_mask=mask.repeat(self.num_heads, 1, 1))

        x, _ = self.self_attention_glu(x)
        x = self.self_attention_glu_add_and_norm(x, enriched)

        # Nonlinear processing on outputs
        decoder = self.decoder_grn(x)
        # Final skip connection
        decoder, _ = self.final_glu(decoder)
        transformer_layer = self.final_glu_add_and_norm(decoder, temporal_feature_layer)
        # Attention components for explainability
        attention_components = {
            # Temporal attention weights
            'decoder_self_attn': self_att,
            # Static variable selection weights
            'static_flags': static_weights[:,:, 0],
            # Variable selection weights of past inputs
            'historical_flags': historical_flags[:, 0, :],
            # Variable selection weights of future inputs
            'future_flags': future_flags[:, 0, :]
        }

        outputs = self.output_layer(transformer_layer[:, self.num_encoder_steps:, :])
        # return outputs, all_inputs, attention_components
        return outputs