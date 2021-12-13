from torch import nn
import torch
from lib.dldl_loss import DLDLLoss
from timm.loss import LabelSmoothingCrossEntropy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_loss(config):
    if config.loss_function == 'dldl_loss':
        return DLDLLoss(lambda1=config.lambda1)
    elif config.loss_function == 'KLDivLoss':
        loss = torch.nn.KLDivLoss(reduction='batchmean')
        return loss
    elif config.loss_function == 'CrossEntropyLoss':
        weight = config.weight.to(device)
        loss = torch.nn.CrossEntropyLoss(weight=weight)
        return loss
    elif config.loss_function == 'LabelSmoothingCrossEntropy':
        loss = LabelSmoothingCrossEntropy(smoothing=0.1)
        return loss
    else:
        print('error loss')
