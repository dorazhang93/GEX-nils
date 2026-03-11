import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F
import math
from torch import Tensor

import typing as ty


def reglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)

# number of parameters and memory
class MLPEncoder(nn.Module):
    """
    Inspired by
    Convolutional autoencoder for genotype data, as described in [1] Kristiina Ausmees, Carl Nettelblad, A deep learning
     framework for characterization of genotype data, G3 Genes|Genomes|Genetics, 2022;, jkac020,
      https://doi.org/10.1093/g3journal/jkac020
    """
    def __init__(self,
                 in_channel: int,
                 in_dim: int,
                 hidden_dim: int,
                 **kwargs):
        super(MLPEncoder,self).__init__()
        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.01),
            nn.Linear(in_dim*in_channel, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=0.01),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
        )

    def forward(self, input: Tensor):
        # input shape: (N,C,L)
        return self.encoder(input)


class Tokenizer(nn.Module):

    def __init__(
        self,
        d_numerical: int,
        d_token: int,
        bias: bool,
    ) -> None:
        super().__init__()
        d_bias = d_numerical

        # take [CLS] token into account
        # self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.weight = nn.Parameter(Tensor(d_numerical, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self) -> int:
        return len(self.weight)

    def forward(self, x_num: Tensor) -> Tensor:
        assert x_num is not None
        # x_num = torch.cat(
        #     [torch.ones(len(x_num), 1, device=x_num.device)]  # [CLS]
        #     + [x_num],
        #     dim=1,
        # )
        x = self.weight[None] * x_num[:, :, None]
        if self.bias is not None:
            # bias = torch.cat(
            #     [
            #         torch.zeros(1, self.bias.shape[1], device=x.device),
            #         self.bias,
            #     ]
            # )
            x = x + self.bias[None]
        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, initialization: str,
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = dropout

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)


    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        mask: Tensor, # batch_size, n_kv_tokens
    ) -> Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0

        batch_size = len(q)
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        v = self._reshape(v)

        if mask is not None:
            assert len(mask.shape)==2
            n_kv = mask.shape[1]
            mask = mask.repeat(1, self.n_heads*n_q_tokens).reshape(-1,self.n_heads, n_q_tokens,n_kv) # (batch_size, self.n_heads,n_q_tokens, n_kv_tokens)
            mask = mask.logical_not()

        x = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=self.dropout)

        x = (
            x.transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x

class Enformer(nn.Module):
    """Transformer.

    References:
    - https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
    - https://github.com/facebookresearch/pytext/tree/master/pytext/models/representations/transformer
    - https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L19
    """

    def __init__(
        self,
        # tokenizer
        in_dim: int,
        d_token: int,
        token_bias: bool,
        # transformer
        n_layers: int,
        n_heads: int,
        d_ffn_factor: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        **kwargs,
    ) -> None:

        super().__init__()
        self.tokenizer = Tokenizer(in_dim, d_token, token_bias)
        n_tokens = self.tokenizer.n_tokens

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization),
                    'linear0': nn.Linear(
                        d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()

            self.layers.append(layer)

        self.activation = reglu
        self.last_activation = F.relu
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x_num: Tensor, isnan_mask: Tensor) -> Tensor:
        # isnan_mask = torch.concat([torch.full((len(x_num), 1), True, device=x_num.device), isnan_mask],
        #                           dim=1)  # concate mask with cls token mask = True
        x = self.tokenizer(x_num).float()


        for layer_idx, layer in enumerate(self.layers):

            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                x_residual,
                x_residual,
                isnan_mask,
            )

            # for computational efficiency only calculate CLS token for the last layer
            # x_residual = layer['attention'](
            #     # for the last attention, it is enough to process only [CLS]
            #     (x_residual[:, :1] if is_last_layer else x_residual),
            #     x_residual,
            #     isnan_mask,
            # )
            # if is_last_layer:
            #     x = x[:, : x_residual.shape[1]]

            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        # assert x.shape[1] == 1
        # x = x[:, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        return x
