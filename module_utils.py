from transformers.models.t5.modeling_t5 import T5Block
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, LayerNorm
from torch import tensor


class AdapterT5Block(T5Block):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.adapter = AdapterBlock(config.d_model, config.adapter_size)

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        output_projected = self.adapter(output[0])
        return (output_projected,) + output[1:]


class AdapterBlock(nn.Module):
    def __init__(self, in_channels, hidden_dimension):
        super().__init__()
        self.norm = LayerNorm(in_channels)
        self.down_projection = Linear(in_channels, hidden_dimension)
        self.activation = ReLU()
        self.up_projection = Linear(hidden_dimension, in_channels)

    def forward(self, x):
        hidden = self.norm(x)
        hidden = self.down_projection(hidden)
        hidden = self.activation(hidden)
        output = self.up_projection(hidden)
        return output + x
