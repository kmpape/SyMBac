import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

class PSFNet(nn.Module):
    def __init__(self, N, output_intensity = None):
        super(PSFNet, self).__init__()
        self.polynomial_weights = nn.Linear(1,1,bias = False)
        self.polynomial_weights.weight.data[0] = -1
        self.actual_intensity = nn.Linear(N,1,bias=False)
        for param in self.actual_intensity.parameters():
            param.requires_grad = False
        if output_intensity is not None:
            for i in range(N):
                self.actual_intensity.weight.data[0][i] = output_intensity[i]
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