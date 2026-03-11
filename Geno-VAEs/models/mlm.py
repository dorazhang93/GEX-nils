from . import BaseVAE
from .module import Encoders
from .module import Decoders
from .loss import masked_mse_loss
import numpy as np
import torch
from torch import nn
from torch import Tensor
from copy import deepcopy

class MLM(BaseVAE):

    def __init__(self,
                 in_dim: int ,
                 d_token: int,
                 encoder: str = "EncoderTransformer",
                 decoder: str = "MLP",
                 mask_scale: float = 0.3,
                 mask_offset: float = 0.05,
                 loss: str = "MSE",
                 **kwargs):
        super(MLM, self).__init__()
        self.mask_scale = mask_scale
        self.mask_offset = mask_offset
        self.in_dim = in_dim
        assert loss == 'MSE'

        print(f"%%% building MLM with {encoder} and {decoder}%%%")

        assert encoder=='EncoderTransformer'
        self.encoder = Encoders[encoder](in_dim,d_token, **kwargs)
        self.decoder = Decoders[decoder](in_dim=d_token,out_dim=in_dim)

    def encode(self, input, isnan_mask):
        x=input.float()
        x = self.encoder(x,isnan_mask)
        return x

    def decode(self,z):
        return self.decoder(z)


    def forward(self, input: Tensor):
        input = torch.clamp(input, min=-4.0, max=6.0)
        missing_placeholder = -1
        device = input.device

        isnan_mask = torch.isnan(input) #  of b,n
        input[isnan_mask] = missing_placeholder
        x=deepcopy(input)

        # random mask
        spar = self.mask_scale * np.random.uniform() + self.mask_offset
        rand_mask = torch.from_numpy((np.random.random_sample(x.shape) < spar)).to(device)

        # input_mask for imputing placeholder and attention mask
        input_mask = rand_mask | isnan_mask
        x[input_mask]= missing_placeholder

        # output_mask for calculating loss
        output_mask = rand_mask & (~isnan_mask)

        z= self.encode(x,input_mask)

        recons = self.decode(z)

        return recons , input, output_mask

    def loss_function(self, args, **kwargs):
        recons = args[0] # (N,C,L)
        y_true =args[1]  # (N,C,L)
        mask = args[2] # True to be added to loss
        recons_loss = masked_mse_loss(recons,y_true,mask)
        return {'loss': recons_loss}