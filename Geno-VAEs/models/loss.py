from torch.nn import functional as F
import torch
import numpy as np


def y_onehot(y_true):
    y_true = F.one_hot((y_true * 2).long(), num_classes=3)
    return y_true

def reconstruction_loss(recons, y_true, loss = "MSE"):
    if loss == "MSE":
        recons_loss = F.mse_loss(y_true,recons, reduction='none')
    else:
        raise NotImplementedError
    recons_loss = recons_loss.mean()
    return recons_loss

def masked_mse_loss(recons, y_true, mask):
    mask = mask.float().to(recons.device)
    recons_loss = F.mse_loss(y_true,recons, reduction='none')
    recons_loss = (recons_loss * mask).sum()
    recons_loss = recons_loss / mask.sum()
    # print('loss', recons_loss)
    return recons_loss

def KLD_loss(latent_dist):
    mu , logvar = latent_dist
    kld_loss = torch.mean(-0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
    return kld_loss

def regulization_loss(z, reg_factor):
    reg_loss = reg_factor * torch.mean(z ** 2)
    return reg_loss


def log_Normal_pdf(z, mu, logv):
    lpdf = -0.5 * (torch.log(2 * np.pi * torch.ones(1).to("cuda:0")) + logv + (z-mu)**2 * torch.exp(-logv))
    return lpdf.mean()

