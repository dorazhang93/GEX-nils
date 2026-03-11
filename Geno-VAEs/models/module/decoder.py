import torch.nn as nn
import torch.nn.init as nn_init
import math
class View(nn.Module):
    def __init__(self, dim,  shape):
        super(View, self).__init__()
        self.dim = dim
        self.shape = shape

    def forward(self, input):
        new_shape = list(input.shape)[:self.dim] + list(self.shape) + list(input.shape)[self.dim+1:]
        return input.view(*new_shape)

class MLPDecoder(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int,
                 out_channel: int,
                 out_dim: int,
                 **kwargs):
        super(MLPDecoder, self).__init__()
        # Build decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.01),
            nn.Linear(hidden_dim, out_dim*out_channel),
            nn.BatchNorm1d(out_dim*out_channel),
            View(1,(out_channel,out_dim)),
        )

    def forward(self, z):
        x = self.decoder(z)
        return x

# class MLP(nn.Module):
#     def __init__(self,
#                  in_dim: int,
#                  out_dim: int,):
#         super(MLP, self).__init__()
#         # Build decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(in_dim, in_dim*2),
#             nn.ELU(),
#             nn.BatchNorm1d(in_dim*2),
#             nn.Dropout(0.01),
#             nn.Linear(in_dim*2, out_dim),
#         )
#
#     def forward(self, z):
#         x = self.decoder(z)
#         return x

class MLP(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,):
        super(MLP, self).__init__()
        # Build decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, in_dim*4),
            nn.ELU(),
            nn.LayerNorm(in_dim*4),
            nn.Dropout(0.01),
            nn.Linear(in_dim*4, 1),
        )

    def forward(self, z):
        x = self.decoder(z).squeeze(-1)
        return x