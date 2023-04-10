import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, ratio = 16) -> None:
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels//ratio, bias=False),
            nn.ReLU(True),
            nn.Linear(channels//ratio, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b,c,_,_ = x.size()
        w = self.squeeze(x).view(b,c)
        w = self.excitation(w).view(b,c,1,1)
        x = x * w
        return x