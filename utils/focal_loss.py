import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.losses import focal

class FocalLoss():
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma=2, alpha=0.5, reduction='none'):
        super().__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, input, target):
        return focal.focal_loss(input, target, gamma = self.gamma, alpha = self.alpha, reduction = self.reduction)