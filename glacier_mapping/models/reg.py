#!/usr/bin/env python
"""
Helpers for Regularizing Models
"""
import torch

def l1_reg(params, lambda_reg, device):
    """
    Compute l^1 penalty for parameters list
    """
    penalty = torch.tensor(0.0).to(device)
    for param in params:
        penalty += lambda_reg * torch.sum(abs(param))
    return penalty


def l2_reg(params, lambda_reg, device):
    """
    Compute l^2 penalty for parameters list
    """
    penalty = torch.tensor(0.0).to(device)
    for param in params:
        penalty += lambda_reg * torch.norm(param, 2) ** 2
    return penalty
