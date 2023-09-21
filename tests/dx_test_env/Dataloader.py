import torch
from utils import GetSourcePts, EuclideanDistance, AverageFilter
import random

class Dataloader(torch.utils.data.Dataset):
    def __init__(self, output, mask):
        """
        Initialise the dataloader
        output: the output image
        mask: the mask of the output image
        """

        sourcePts = GetSourcePts(mask)
        (outputWidth, outputHeight) = output.shape
        N = len(sourcePts)
        distanceMatrix= []
        intensityMatrix = []
        maskSize = EuclideanDistance(mask.shape,(0,0)) 
        maxOutput = max(output.flatten())

        for idx in range(N):
            current = sourcePts[idx]
            distance = [EuclideanDistance(sourcePts[k],current)/maskSize for k in range(N)] #Normalise by diagonal length of mask
            intensty = [AverageFilter(output,current[0],current[1])/maxOutput] #Normalise by max intensity
            distanceMatrix.append(torch.tensor(distance))
            intensityMatrix.append(torch.tensor(intensty))

        """
        #Add a bunch of random points
        for i in range(outputWidth*outputHeight):
            x = random.randint(0,maskWidth-1)
            y = random.randint(0,maskHeight-1)
            distance = [EuclideanDistance(sourcePts[k],(x,y))/maskSize for k in range(N)] #Normalise by diagonal length of mask
            intensty = [AverageFilter(out,x,y)/65535] #Normalise by max intensity
            distanceMatrix.append(torch.tensor(distance))
            intensityMatrix.append(torch.tensor(intensty))
        """

        #Convert to tensor
        distanceMatrix = torch.stack(distanceMatrix)
        intensityMatrix = torch.stack(intensityMatrix)

        self.distanceMatrix = distanceMatrix
        self.intensityMatrix = intensityMatrix
    
    def __len__(self):
        """
        Get the number of samples in the dataset
        """
        return len(self.intensityMatrix)

    def __getitem__(self, idx):
        """
        Get the sample at index idx
        """
        return (self.distanceMatrix[idx], self.intensityMatrix[idx]) 