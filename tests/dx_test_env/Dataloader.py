import torch
import torch.utils.data
from utils import GetSourcePts, EuclideanDistance, AverageFilter
import random

class Dataloader(torch.utils.data.Dataset):
    def __init__(self, output, mask, randomPts=0, adjPts=0):
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

        """
        for idx in range(N):
            current = sourcePts[idx]
            distance = [EuclideanDistance(sourcePts[k],current)/maskSize for k in range(N)] #Normalise by diagonal length of mask
            intensty = [AverageFilter(output,current[0],current[1])/maxOutput] #Normalise by max intensity
            distanceMatrix.append(torch.tensor(distance))
            intensityMatrix.append(torch.tensor(intensty))
        """
        #Add a bunch of random points
        for i in range(randomPts):
            x = random.randint(0,outputWidth-1)
            y = random.randint(0,outputHeight-1)
            distance = [EuclideanDistance(sourcePts[k],(x,y))/maskSize for k in range(N)] #Normalise by diagonal length of mask
            intensty = [AverageFilter(output,x,y)/maxOutput] #Normalise by max intensity
            distanceMatrix.append(torch.tensor(distance))
            intensityMatrix.append(torch.tensor(intensty))
        
        #Add adjacent points including the source points
        for (x,y) in sourcePts:
            for i in range(-adjPts,adjPts+1):
                for j in range(-adjPts,adjPts+1):
                    if x+i >= 0 and x+i < len(output) and y+j >= 0 and y+j < len(output[0]):
                        distance = [EuclideanDistance(sourcePts[k],(x+i,y+j))/maskSize for k in range(N)]
                        intensty = [AverageFilter(output,x+i,y+j)/maxOutput]
                        distanceMatrix.append(torch.tensor(distance))
                        intensityMatrix.append(torch.tensor(intensty))

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