import torch
import torch.nn as nn

class ESRLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(ESRLoss, self).__init__()
        self.eps = eps

    def forward(self, prediction, target):
        # prediction, target shape: [Batch, Length, 1]
        error = torch.sum(torch.square(target - prediction), dim=1)
        signal = torch.sum(torch.square(target), dim=1)
        
        # Error-to-Signal Ratio
        esr = error / (signal + self.eps)
        
        return torch.mean(esr)

class DCLoss(nn.Module):
    def __init__(self):
        super(DCLoss, self).__init__()

    def forward(self, prediction, target):
        prediction_mean = torch.mean(prediction, dim=1)
        target_mean = torch.mean(target, dim=1)
        return torch.mean(torch.square(target_mean - prediction_mean))

class LSTMCombinedLoss(nn.Module):
    def __init__(self):
        super(LSTMCombinedLoss, self).__init__()
        self.esr = ESRLoss()
        self.dc = DCLoss()

    def forward(self, prediction, target):
        return self.esr(prediction, target) + self.dc(prediction, target)
