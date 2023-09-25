import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

class FirstOrderBesselApprox(nn.Module):
    def __init__(self):
        super(FirstOrderBesselApprox, self).__init__()
        self.offset = nn.Parameter(requires_grad=True)
    def getPSF(self,x):
        stack = []
        for i in range (1, 1+1):
            stack.append(torch.pow(x,2*i))
        x = torch.stack(stack, dim=2)
        x = self.polynomial_weights(x)
        x = x.reshape(x.shape[0],x.shape[1])
        x = torch.exp(x)
        x = x * (- math.pi/(2*self.polynomial_weights.weight.data))**0.5
        return x
        
    def forward(self, x):
        x = self.getPSF(x)
        x = self.actual_intensity(x)
        return x