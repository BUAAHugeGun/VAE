import torch
import torch.nn as nn
from loss.SSIM import MSSSIM


def vae_loss(x, y, mu, log_var, lambdas: dict):
    kld_loss = torch.mean(-0.5 * (torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)), dim=0) * lambdas['kld']
    loss = kld_loss
    losses = {'kld': kld_loss}
    if lambdas.get('mse') is not None:
        losses['mse'] = lambdas['mse'] * nn.MSELoss()(x, y)
        loss += losses['mse']
    if lambdas.get('l1') is not None:
        losses['l1'] = lambdas['l1'] * nn.L1Loss()(x, y)
        loss += losses['l1']
    if lambdas.get('msssim') is not None:
        losses['msssim'] = lambdas['msssim'] * MSSSIM()(x, y)
        loss += losses['msssim']
    losses['total'] = loss
    return losses
