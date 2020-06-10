#!/usr/bin/env python
"""
Helpers for Regularizing Models
"""
import numpy as np

def l1_reg(params, lambda_reg, device):
    penalty = torch.tensor(0.0).to(device)
    for param in params:
        penalty += lambda_reg * torch.sum(abs(param))

    return penalty


def l2_reg(params, lambda_reg, device):
    penalty = torch.tensor(0.0).to(device)
    for param in params:
        penalty += lambda_reg * torch.norm(param, 2) ** 2

    return penalty
