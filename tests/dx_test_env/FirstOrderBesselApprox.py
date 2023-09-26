import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from scipy.special import jv

class FirstOrderBesselApprox(nn.Module):
    def __init__(self):
        super(FirstOrderBesselApprox, self).__init__()
        self.offset = nn.Parameter(requires_grad=False)
        self.bessel_weight = nn.Parameter(requires_grad=True)
    def getPSF(self,x):
        x = x * self.bessel_weight
        x = (2*torch.special.bessel_j1(x) / x)**2
        x = x + self.offset
        return x
        
    def forward(self, x):
        x = self.getPSF(x)
        return x