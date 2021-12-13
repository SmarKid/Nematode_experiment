import torch
import torch.nn as nn


def kl_loss(inputs, labels):
    criterion = nn.KLDivLoss(reduction='batchmean')
    # outputs = torch.log(inputs)
    loss = criterion(inputs, labels)
    return loss


def L1_loss(inputs, labels):
    criterion = nn.L1Loss(reduction='mean')
    loss = criterion(inputs, labels.float())
    return loss


def dldl_loss(y_hat, y, ld_hat, ld, lambda1):
    loss1 = kl_loss(ld_hat, ld)
    loss2 = L1_loss(y_hat, y)
    total_loss = loss1 + lambda1 * loss2
    return total_loss


class DLDLLoss:
    def __init__(self, lambda1=1):
        self.lambda1 = lambda1
        pass

    def __call__(self, y_hat, y, ld_hat, ld):
        """
            y_hat：预测标签
            y: 真实标签
            ld_hat: 预测分布
            ld: 真实分布
        """
        return dldl_loss(y_hat, y, ld_hat, ld, self.lambda1)
