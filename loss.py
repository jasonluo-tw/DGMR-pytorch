import torch
import torch.nn.functional as F
import torch.nn as nn

def hinge_loss_dis(fake_score, real_score):
    """
    hinge loss for discriminator
    """
    l1 = torch.mean(F.relu(1.0 - real_score))
    l2 = torch.mean(F.relu(1.0 + fake_score))

    loss = l1 + l2

    return loss


def hinge_loss_gen(fake_score):
    """
    hinge loss for generator
    """
    loss = -torch.mean(fake_score)

    return loss

def grid_cell_regularizer(fake_samples, targets):
    """
    regularize for grid cell to let the prediction closer to the ground truth
    Monte Carlo estimates for expectations
    fake_samples -> (n, batch, 18, 1, 256, 256)
    batch_targets -> (batch, 18, 1, 256, 256)
    
    Return:
        loss -> (batch)
    """
    fake_mean = torch.mean(fake_samples, dim=0)
    weights = torch.clip(targets, 0.0, 24.0)
    loss = torch.mean(torch.abs(fake_mean - targets) * weights)

    return loss

