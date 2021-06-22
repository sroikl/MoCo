"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""

import torch
from torch import nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.tau = temperature

    def forward(self, logits, l_pos, p):

        # Compute contrastive loss:
        l_pos = torch.exp(l_pos / self.tau)
        logits = torch.exp(logits / self.tau)
        # l_neg = torch.exp(torch.mul(logits, neg_mask.float()) / self.tau)
        loss = (torch.log(logits.sum(1)) - torch.log(l_pos.sum(1))) / p

        return loss.mean()
