---
title: Classic LLM Models and Architectures
date: 2025-01-20 09:00:00 +0800
categories: [Machine Learning, LLM]
tags: [llm, transformer, deep-learning, ai]
image: /assets/img/post/llm_arch/llm_arch1.png  # Optional: Add if you want a header image
toc: true                  # Optional: Table of Contents
math: true               # Optional: Set to true for mathematical notations
mermaid: true           # Optional: Set to true for architecture diagrams
---

# Introduction

Brief overview of Large Language Models (LLMs) and their architectural evolution.

## Model Architectures

### Transformer Architecture
- Original transformer design
- Self-attention mechanism
- Multi-head attention
- Position embeddings
- Feed-forward networks

### GPT Architecture
- Decoder-only transformer
- Autoregressive modeling
- Scaled attention
- Layer normalization
- Architectural improvements across versions

### BERT Architecture
- Encoder-only transformer
- Bidirectional attention
- Masked language modeling
- Next sentence prediction
- Pre-training objectives

##### Architecture Details
- Encoder-only transformer architecture
- Bidirectional context modeling
- Multiple transformer layers with:
  - Multi-head self-attention
  - Feed-forward networks
  - Layer normalization
  - Residual connections

##### Embedding System
- Three types of embeddings combined:
  1. Word Embeddings (word_embeddings)
     - Maps vocabulary tokens to dense vectors
     - Learned representations of input tokens
  2. Position Embeddings (position_embeddings)
     - Encodes position information
     - Learned embeddings for each position
     - Maximum sequence length defined by config
  3. Token Type Embeddings (token_type_embeddings)
     - Distinguishes between different segments
     - Used for sentence pair tasks
     - Enables multi-sentence modeling
- Combined through addition
- Layer normalization and dropout applied

##### Attention Mechanism
- Multi-head self-attention
- Components:
  - Query (Q): Current token's representation
  - Key (K): All tokens' matching features
  - Value (V): All tokens' content features
- Bidirectional attention:
  - Each token can attend to all other tokens
  - No masking in encoder
  - Full contextual understanding

##### Activation Function
- Uses GELU (Gaussian Error Linear Unit)
- Smoother alternative to ReLU
- Properties:
  - Non-linear activation
  - Approximates multiplicative noise
  - Better gradient properties
  - Standard choice in modern transformers

##### Layer Normalization
- Applied after attention and feed-forward layers
- Normalizes input across feature dimension
- Features:
  - Mean and variance normalization
  - Learnable scale and bias
  - Improves training stability
  - Better gradient flow

##### Key Implementation

```python
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
```

### RoBERTa
- Improved BERT training methodology
- Dynamic masking
- Removal of Next Sentence Prediction (NSP)
- Larger batch sizes
- Longer training with more data

### T5 (Text-to-Text Transfer Transformer)
- Encoder-decoder architecture
- Unified text-to-text framework
- Comprehensive pre-training objectives
- Flexible task formatting
- Strong multi-task performance

##### Architecture Details
- Complete encoder-decoder transformer architecture
- Unified approach to all NLP tasks:
  - Converts every task into text-to-text format
  - Input text → Transformer → Output text
  - Enables multi-task learning and transfer
- Modular design:
  - Encoder processes input text
  - Decoder generates output text
  - Cross-attention connects encoder and decoder

##### Attention Mechanism
- Three types of attention:
  - Encoder self-attention (bidirectional)
    - Each token attends to all input tokens
    - Full contextual understanding
  - Decoder masked self-attention
    - Autoregressive generation
    - Prevents future information leakage
  - Cross-attention
    - Decoder attends to encoder outputs
    - Connects input understanding to output generation

##### Layer Normalization
- Uses RMSNorm (Root Mean Square Layer Normalization)
- Key characteristics:
  - Only scales, no shift operation
  - Variance calculated without mean
  - No bias terms
- Mathematical formulation:
  $$
  \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}} \cdot \gamma
  $$
- Benefits:
  - Improved training stability
  - Reduced computational complexity
  - Better numerical properties

##### Activation Function
- Uses GELU (Gaussian Error Linear Unit) with "new" approximation
- Similar to GPT-2's activation function
- Smooth transition between active and inactive states
- Mathematical form:
  $$
  \text{GELU}(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))
  $$

##### Core Components

```python

class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states

class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states

class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

```

###### Attention and Block Implementation
```python
class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs

class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs
```

Key Features:
- RMSNorm-style layer normalization (scale-only)
- Two feed-forward variants:
  - Standard dense-act-dense
  - Gated version with parallel transformations
- Flexible block structure supporting both encoder and decoder
- Relative position bias in attention
- Optional cross-attention for decoder blocks
- Residual connections and dropout throughout

### GPT-2
- Improved scaling from original GPT
- Larger context window (1024 tokens)
- Layer normalization adjustments
- Zero-shot task learning
- WebText training corpus

##### Architecture Details
- Decoder-only transformer architecture
- Autoregressive language modeling
- Deep neural network with multiple transformer blocks
- Each block contains:
  - Multi-head self-attention layer
  - Feed-forward neural network
  - Layer normalization and residual connections

##### Attention Mechanism
- Multi-head causal self-attention
- Components:
  - Query (Q): What the current token is looking for
  - Key (K): What each token offers for matching
  - Value (V): The actual information to be aggregated
- Masked attention pattern:
  - Each token can only attend to previous tokens
  - Ensures autoregressive property
  - Prevents information leakage from future tokens

##### Activation Function
- Uses GELU (Gaussian Error Linear Unit) with "new" approximation
- Smoother alternative to ReLU
- Mathematical form:
  $$
  \text{GELU}(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))
  $$
- Benefits:
  - Better gradient flow
  - More natural probabilistic interpretation
  - Improved training dynamics

##### Layer Normalization
- Applied before attention and feed-forward layers
- Normalizes input across feature dimension
- Key components:
  - Mean and variance normalization
  - Learnable scale and bias parameters
- Helps with:
  - Training stability
  - Faster convergence
  - Better gradient flow

##### Implementation Details

###### MLP Block

```python
class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function] # select act function 
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
```

###### Attention Implementation

```python
class GPT2Attention(nn.Module):
 def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.pruned_heads = set()

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights
```

###### Transformer Block

```python
class GPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs
```

Key Implementation Features:
- Pre-layer normalization architecture
- Residual connections around both attention and MLP blocks
- Causal masking for autoregressive prediction
- Configurable attention scaling and dropout
- Support for cross-attention in conditional generation tasks 

### LLaMA Family

#### LLaMA 1
- Efficient scaling techniques
- Pre-normalization using RMSNorm
- SwiGLU activation function
- Rotary positional embeddings
- Open-weights approach

##### Architecture Details
- Decoder-only transformer architecture (similar to PaLM)
- Each token can only attend to itself and previous tokens
- Key architectural improvements over standard transformer:
  - SwiGLU activation function replaces ReLU for better performance
  - RMSNorm (Root Mean Square) layer normalization for improved training stability
  - RoPE (Rotary Positional Embeddings) for better position encoding
- Optimized for compute efficiency while maintaining performance

##### Embedding System
- Uses Rotary Positional Embeddings (RoPE)
- Two implementations:
  1. Standard RoPE (LlamaRotaryEmbedding):
     - Enables relative position modeling
     - Rotation-based position encoding
     - Better handling of variable-length sequences
  2. Linear Scaling RoPE (LlamaLinearScalingRotaryEmbedding):
     - Extends context window linearly
     - Maintains position sensitivity at longer distances
     - Improved extrapolation beyond training length

##### Rotary Embedding Implementation

```python
class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
```

##### Attention Mechanism
- Multi-head self-attention with causal masking
- Sliding window attention for efficiency
- Key features:
  - Scaled dot-product attention
  - RoPE integration for position-aware attention
  - Optimized for autoregressive generation

##### Layer Normalization
- Uses RMSNorm (equivalent to T5LayerNorm)
- Characteristics:
  - No mean subtraction (unlike traditional LayerNorm)
  - Only scales inputs based on root mean square
  - No bias terms
- Benefits:
  - Reduced computational complexity
  - Better training stability
  - Improved convergence

##### Activation Function
- Uses SiLU (Sigmoid Linear Unit) activation
- Also known as Swish activation
- Mathematical form:
  $$
  \text{SiLU}(x) = x \cdot \sigma(x)
  $$
  where σ(x) is the sigmoid function
- Historical context:
  - Originally introduced in GELU paper (Hendrycks et al.)
  - Further developed in reinforcement learning (Elfwing et al.)
  - Popularized as Swish (Ramachandran et al.)
- Advantages:
  - Smooth gradient flow
  - Non-monotonic behavior
  - Better performance in deep networks

##### RMSNorm Details
- Simplified version of LayerNorm that focuses only on re-scaling
- Removes mean-centering operation from traditional LayerNorm
- Key advantages:
  - Computationally more efficient
  - Maintains model stability through re-scaling invariance
  - Reduces training complexity while preserving performance
- Mathematical formulation:
  $$
  \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}} \cdot \gamma
  $$
- Hypothesis: Re-scaling invariance, not re-centering, is the key factor for normalization success

##### PaLM Similarities
- Shares core decoder-only transformer architecture
- Similar autoregressive prediction approach
- Both focus on scaling efficiency
- Key difference: LLaMA's architectural choices (RMSNorm, RoPE) vs PaLM's standard components

#### LLaMA 2
- Extended context length (4k tokens)
- Grouped-query attention
- Enhanced instruction tuning
- Improved safety measures
- Better reasoning capabilities

##### Architecture Improvements
- Key differences from LLaMA 1:
  - Extended context window from 2048 to 4096 tokens
  - Grouped-Query Attention (GQA) in 34B and 70B models
    - Improves inference efficiency
    - Better memory usage
    - Maintains model quality while reducing computational cost
  - Retains core architectural elements (RMSNorm, RoPE, SwiGLU)

##### Grouped-Query Attention (GQA)
- Novel attention mechanism that improves inference efficiency
- Core concept:
  - Groups query heads into subgroups
  - Each subgroup shares a single key and value head
  - Reduces computation while maintaining attention quality
- Benefits:
  - Significantly reduces memory usage
  - Improves inference speed
  - Maintains model performance
- Technical details:
  - Traditional attention: Each query head has its own key and value heads
  - GQA: Multiple query heads share common key and value heads
  - Reduces parameter count and memory footprint
- Motivation:
  - Query heads often perform similar attention patterns
  - Sharing key-value pairs exploits this redundancy
  - Particularly effective in larger models (34B and 70B)

##### Training Innovations
- Enhanced training methodology:
  - Rejection Sampling fine-tuning
    - Generates K candidate outputs per prompt
    - Uses reward model to select best response
    - Improves output quality and safety
  - Supervised Fine-Tuning (SFT)
  - Iterative fine-tuning with human feedback
- Focused on:
  - Safety and alignment
  - Response quality
  - Instruction following
  - Reduced toxicity

#### LLaMA 3
- Advanced context processing
- Multimodal capabilities
- Enhanced reasoning
- Improved factual accuracy
- Reduced hallucination rates

### BLOOMZ
- Multilingual architecture
- Prompt-based cross-lingual transfer
- Zero-shot cross-lingual generalization
- Multitask prompted training
- Culture-aware responses

##### ALiBi (Attention with Linear Biases)
- Alternative to positional embeddings
- Key characteristics:
  - Adds linear bias to attention scores
  - Position information through attention bias
  - No learned position embeddings
- Implementation:
```python
def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)
```

Key Features:
- Advantages:
  - Extrapolates to longer sequences
  - No maximum sequence length limit
  - Parameter-free position encoding
- Mathematical formulation:
  $$
  \text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}} + m \cdot p)V
  $$
  where:
  - m is the slope for each attention head
  - p is the relative position matrix
- Implementation details:
  - Different slope for each attention head
  - Slopes decrease exponentially
  - Linear bias grows with sequence distance

References:
- [Theory and Implementation Details (Chinese)](https://kexue.fm/archives/9431)
- [ALiBi Paper: Train Short, Test Long](https://arxiv.org/abs/2108.12409)

### Mixtral
- Mixture of Experts (MoE) architecture
- Sparse activation patterns
- Efficient compute utilization
- State-of-the-art performance
- Balanced parameter-compute trade-off

##### Architecture Details
- Sparse Mixture-of-Experts (SMoE) design:
  - 8 expert feed-forward networks per layer
  - Only top-2 experts activated per token
  - Shared self-attention layers
  - Routing network determines expert selection
- Core components:
  - Transformer backbone
  - Expert modules
  - Gating network
  - Load balancing mechanism

##### Mix-Of-Experts Implementation
```python
class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.args = moe_args

    def forward(self, inputs: torch.Tensor):
        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.args.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs[batch_idx]
            )
        return results
```

Key Features:
- Sparse expert selection:
  - Only activates 2 experts per token
  - Reduces computational cost
  - Maintains model capacity
- Load balancing:
  - Distributes tokens across experts
  - Prevents expert collapse
  - Optimizes resource utilization
- Expert gating:
  - Learned routing mechanism
  - Dynamic expert selection
  - Conditional computation
- Advantages:
  - Better parameter efficiency
  - Reduced inference cost
  - Specialized expert knowledge
  - Improved task performance


## Model Comparison

| Model | Parameters | Context Window | Architecture Type | Key Features |
|-------|------------|----------------|------------------|--------------|
| BERT  | 340M      | 512 tokens     | Encoder-only     | Bidirectional |
| RoBERTa | 355M    | 512 tokens     | Encoder-only     | Dynamic masking |
| T5    | 220M-11B  | 512 tokens     | Encoder-decoder  | Text-to-text |
| GPT-2 | 1.5B      | 1024 tokens    | Decoder-only     | Zero-shot |
| LLaMA 1 | 7B-65B  | 2048 tokens    | Decoder-only     | Efficient scaling |
| LLaMA 2 | 7B-70B  | 4096 tokens    | Decoder-only     | Group attention |
| BLOOMZ | 176B     | 2048 tokens    | Decoder-only     | Multilingual |
| Mixtral | 46.7B   | 32k tokens     | MoE              | Sparse experts |

## Conclusion

Summary of current trends and future directions in LLM architectures.

## References

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
3. [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
