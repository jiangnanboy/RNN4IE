import torch
import torch.nn.functional as F
from torch import nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, size_average=True, ignore_index=-100): # gamma : [0, 5]
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

def define_loss_fce(PAD_IDX=None):
    '''
    define focal loss
    :param PAD_IDX:
    :return:
    '''
    criterion = FocalLoss(ignore_index=PAD_IDX)
    return criterion

def define_loss_ce(PAD_IDX=None):
    '''
    define loss function CE
    :param PAD_IDX:
    :return:
    '''
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    return criterion

def define_loss_bce():
    '''
    define loss function BCE
    :return:
    '''
    criterion = nn.BCELoss()
    return criterion

def define_loss_bcelogits():
    '''
    define loss function BCELogit
    :return:
    '''
    criterion = nn.BCEWithLogitsLoss()
    return criterion