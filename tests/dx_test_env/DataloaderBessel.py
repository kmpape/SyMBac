import torch
import torch.utils.data
import random
import numpy as np

class DataloaderBessel(torch.utils.data.Dataset):
    def __init__(self,psf):
        psfCentre = (psf.shape[0]//2, psf.shape[1]//2)

        distance = np.zeros(len(psf))*len(psf[0])
        magnitude = np.zeros(len(psf))*len(psf[0])

        for i in range(len(psf)):
            for j in range(len(psf[0])):
                distance[i*len(psf[0])+j] = np.linalg.norm(np.asarray((i,j))-np.asarray(psfCentre))
                magnitude[i*len(psf[0])+j] = psf[i][j]

        self.distance = torch.tensor(distance)
        self.magnitude = torch.tensor(magnitude)
    
    def __len__(self):
        return len(self.distance)
    
    def __getitem__(self,idx):
        return self.distance[idx],self.magnitude[idx]