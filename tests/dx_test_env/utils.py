import pickle
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from DataloaderBessel import DataloaderBessel
from FirstOrderBesselApprox import FirstOrderBesselApprox
from scipy.special import jv
from scipy.optimize import curve_fit

def GetSourcePts(mask, filePath=None):
    """
    Get the source points from the mask and save them to a file
    :param mask: mask to get the source points from
    :param filePath: path to save the source points to
    :return: the source points
    """
    sourcePts = np.where(mask != 0)
    sourcePts = np.transpose(sourcePts)
    if filePath is not None:
        with open(filePath + '/sourcePts.pkl', 'wb') as file: 
            pickle.dump(sourcePts, file)

    return np.asarray(sourcePts)

def GetMidPts(mask, filePath=None):
    """
    Set the mid points of each mask and save them to a file
    :param mask: mask to get the mid points from
    :param filePath: path to save the mid points to
    :return: the mid points
    """
    
    maskUsed = []
    midPts = []
    (maskWidth, maskHeight) = mask.shape

    for i in range(maskWidth):
        for j in range(maskHeight):
            if (mask[i,j] not in maskUsed and mask[i,j] != 0):
                indices = np.where(mask == mask[i,j])
                midPts.append((int(np.mean(indices[0])), int(np.mean(indices[1]))))
                maskUsed.append(mask[i,j])

    if filePath is not None:
        with open(filePath + '/midPts.pkl', 'wb') as file: 
            pickle.dump(midPts, file)

    return np.asarray(midPts)

def AverageFilter(img, x, y, sz = 3):
    """
    Applies a bluring filter of size sz centred on (x,y) to the image img
    :param img: image to be filtered
    :param x: x coordinate of the centre of the filter
    :param y: y coordinate of the centre of the filter
    :param sz: size of the filter
    :return: the average value of the pixels in the filter
    """
    assert(sz%2 == 1)
    total = 0
    for i in range(-int(sz/2),int(sz/2)+1):
        for j in range(-int(sz/2),int(sz/2)+1):
            if x+i >= 0 and x+i < len(img) and y+j >= 0 and y+j < len(img[0]):
                total += img[x+i][y+j]
    total/=sz*sz
    return total

def GetPSFMatrix(origin,target,psf):
    """
    Get the point spread function magnitude at target given the origin and the psf in matrix form
    :param origin: origin of the psf
    :param target: target of the psf
    :param psf: psf in matrix form
    :return: the magnitude of the psf at target
    """
    assert(len(psf)%2 == 1) and (len(psf[0])%2 == 1)
    centre = [int(len(psf)/2),int(len(psf[0])/2)]
    shifted_target = [target[0]-origin[0]+centre[0],target[1]-origin[1]+centre[1]]
    if (shifted_target[0] < 0 or shifted_target[0] >= len(psf)) or (shifted_target[1] < 0 or shifted_target[1] >= len(psf[0])):
        return 0
    else:
        return psf[shifted_target[0]][shifted_target[1]]

def GetPSFModel(origin, target, model, diagonal):
    """
    Get the point spread function magnitude at the target given the origin and the psf in model form
    :param origin: origin of the psf
    :param target: target of the psf
    :param model: our trained model
    :param diagonal: diagonal of the image
    :return: the magnitude of the psf at target
    """
    return model.getPSF(torch.tensor([[np.linalg.norm(np.asarray(origin)-np.asarray(target))/diagonal]])).detach().numpy()[0]
    
def PlotPSF(model,psf=None):
    """
    Plot the psf of the model and the real psf (optional)
    :param model: our trained model
    :param psf: actual psf
    :return: None
    """
    if psf is None:
        psf = model.getPSF(torch.tensor([np.linspace(0,1,100,dtype=np.float32)]))
    psf_x = [np.linspace(0,1,100,dtype=np.float32)]
    psf_y = psf.detach().numpy().flatten()
    plt.plot(psf_x,psf_y)
    plt.show()

def GetAvgIntensityWithMask(img, mask, sourcePts=None):
    
    """
    Get Avg Intensity from the image with the mask
    :param img: image to get the average intensity from
    :param mask: mask to get the average intensity from
    :param sourcePts: source points of the mask
    :return: the average intensity of the image with the mask
    """

    if (sourcePts is None):
        sourcePts = GetSourcePts(mask)
    
    avg_intensity = {}
    for x in sourcePts:
        if not mask[x[0],x[1]] in avg_intensity:
            avg_intensity[mask[x[0],x[1]]] = [img[x[0],x[1]],1]
        else:
            avg_intensity[mask[x[0],x[1]]][0] += img[x[0],x[1]]
            avg_intensity[mask[x[0],x[1]]][1] += 1

    mask_value = avg_intensity.keys()
    avg_intensity = [v[0]/v[1] for k, v in avg_intensity.items() ]
    avg_intensity = np.array(avg_intensity)
    avg_intensity = avg_intensity/max(avg_intensity)

    return mask_value, avg_intensity


def TrainingLoop(model,dataloader,lossCriterion,optimizer,epochs=50,savePath=None,displayGraph=False,minLoss=0):
    """
    This is our main training loop for our PSF Net
    :param model: model to be trained
    :param dataloader: dataloader to be used for training
    :param lossCriterion: loss criterion to be used for training
    :param optimizer: optimizer to be used for training
    :param epoch: number of epochs to train for
    :param savePath: path to save the model to
    :param displayGraph: whether to display the graph of the psf
    :param minLoss: minimum loss to stop training
    :return: None
    """
    
    average_running_loss = 0.0

    current_epoch = 0
    while current_epoch < epochs:  # loop over the dataset multiple times
        current_epoch += 1
        average_running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, outputs = data
            nn_outputs = model(inputs)
            loss = lossCriterion(nn_outputs.float(), outputs.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            average_running_loss += loss.item()
        average_running_loss /= len(dataloader)

        if average_running_loss < minLoss:
            break

        if (current_epoch%20==0):
            print("Epoch: ",current_epoch, " Loss: ",average_running_loss)
            if (displayGraph):
                #Normalise x axis of original psf to 1
                psf_x = [np.linspace(0,1,100,dtype=np.float32)]
                psf_y = model.getPSF(torch.tensor(psf_x))
                psf_x = psf_x[0]
                psf_y = psf_y.detach().numpy().flatten()
                plt.plot(psf_x,psf_y)
                plt.show()
                    
    print("Epoch: ",current_epoch, " Loss: ",average_running_loss)
    if savePath is not None:
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_running_loss,
            }, savePath)
        
def InverseMatrix(originalOutput, mask, sourcePts, learningRate = 0.5, randomPts = 0, adjPts = 0, psf=None, model=None):
    """
    Performs Matrix Inversion to get the originalOutput image with the source pts contribution.
    sourcePts with the same mask value are forced to have the same intensity
    Returns a new image based on the originalOuput*(1-learningRate) + learningRate*newOutput

    :param originalOutput: original output image
    :param mask: mask of the original output image
    :param sourcePts: source points of the original output image
    :param GetPSF: selected method to get PSF
    :param learningRate: learning rate of the matrix inversion
    :param randomPts: number of random points to be included in the matrix inversion
    :param adjPts: number of adjacent points to be included in the matrix inversion
    :param psf: psf to be used for the matrix inversion
    :param model: model to be used for the matrix inversion
    :return: new image based on the originalOuput*(1-learningRate) + learningRate*newOutput
    """

    assert(psf is not None or model is not None)

    additionalPts = []
    """
    #Include points along the edge
    for i in range(len(originalOutput)):
        additionalPts.append([i,0])
        additionalPts.append([i,len(originalOutput[0])-1])
    for i in range(len(originalOutput[0])):
        additionalPts.append([0,i])
        additionalPts.append([len(originalOutput)-1,i])
    """

    #Include random points
    for i in range(randomPts):
        additionalPts.append([np.random.randint(0,len(originalOutput)),np.random.randint(0,len(originalOutput[0]))])

    """
    #Add adjacent points including the source points (allow repeats)
    for (x,y) in sourcePts:
        for i in range(-adjPts,adjPts+1):
            for j in range(-adjPts,adjPts+1):
                if x+i >= 0 and x+i < len(originalOutput) and y+j >= 0 and y+j < len(originalOutput[0]):
                    additionalPts.append([x+i,y+j])
    """

    #Add adjacen points including the source pts (no repeats)
    chosenPts = np.zeros((len(mask),len(mask[0])))
    for (x,y) in sourcePts:
        for i in range(-adjPts,adjPts+1):
            for j in range(-adjPts,adjPts+1):
                if x+i >= 0 and x+i < len(originalOutput) and y+j >= 0 and y+j < len(originalOutput[0]):
                    chosenPts[x+i,y+j] = 1
    
    for x in range(len(chosenPts)):
        for y in range(len(chosenPts[0])):
            if chosenPts[x,y] == 1:
                additionalPts.append([x,y])

    #Count number of unique masked points
    maskedPts = []
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i,j] not in maskedPts and mask[i,j] != 0:
                maskedPts.append(mask[i,j])

    originalOutputSize = np.linalg.norm([len(originalOutput),len(originalOutput[0])])
    print(maskedPts)

    A = np.zeros((len(additionalPts), len(maskedPts)))
    if psf is not None:
        #Generate the matrix A
        for i, x in enumerate(additionalPts):
            for j, y in enumerate(sourcePts):
                A[i, maskedPts.index(mask[y[0]][y[1]])] += GetPSFMatrix(y,x,psf)
    if model is not None:
        #Generate the matrix A
        for i, x in enumerate(additionalPts):
            for j, y in enumerate(sourcePts):
                A[i, maskedPts.index(mask[y[0]][y[1]])] += GetPSFModel(y,x,model,originalOutputSize)
    #Generate the vector b
    b = np.zeros((len(additionalPts), 1))
    for i, (x, y) in enumerate(additionalPts):
        b[i] = AverageFilter(originalOutput, x, y)
    b = b / max(originalOutput.flatten())

    #Solve the matrix
    x_bar = np.linalg.lstsq(A, b,rcond=None)
    x_bar = x_bar[0]
    x_bar = x_bar/max(x_bar)
    print(x_bar)

    #Generate the new image
    newOutput = np.zeros((len(originalOutput), len(originalOutput[0])))
    for i, (x, y) in enumerate(sourcePts):
        newOutput[x, y] = x_bar[maskedPts.index(mask[x][y])]

    return originalOutput*(1-learningRate) + learningRate*newOutput

def ApproxPSFBesselModel(psf, trainingEpochs = 100):
    """
    Calculate PSF that best fits our inverse PSF from the model
    :param psf: psf to be approximated
    :param trainingEpochs: number of epochs to train for
    :return: the approximated psf
    """
    data = DataloaderBessel(psf)
    dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True, num_workers=0)

    model = FirstOrderBesselApprox()
    model.offset.data = torch.tensor([0.0])
    model.bessel_weight.data = torch.tensor([100.0])
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.5)

    for e in range(trainingEpochs):
        average_running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, outputs = data
            nn_outputs = model(inputs)
            loss = criterion(nn_outputs.float(), outputs.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            average_running_loss += loss.item()
        average_running_loss /= len(dataloader)
        print("Epoch: ",e, " Loss: ",average_running_loss)
        print(model.bessel_weight.data, model.offset.data)
    
    newPSF = np.zeros((len(psf),len(psf[0])))
    for i in range(len(newPSF)):
        for j in range(len(newPSF[0])):
            newPSF[i,j] = model.getPSF(torch.tensor([[np.linalg.norm(np.asarray((i,j))-np.asarray((len(psf)//2,len(psf[0])//2)))/np.linalg.norm(np.asarray((len(psf)//2,len(psf[0])//2)))]])).detach().numpy()[0]

    newPSF[len(psf)//2,len(psf[0])//2] = 1
    return newPSF

def ApproxPSFBesselOptimise(psf, cutoff = 0):
    """
    Calculate PSF that best fits our inverse PSF from the oprimisation
    :param psf: psf from inverse fourier
    :param cutoff: cutoff magnitude for consideration in the inverse matrix
    :return: the approximated psf
    """

    psf = psf/np.max(psf)

    def funcPSF(x,a):
        return (2*jv(1,x*a) / (x*a))**2

    xData = []
    yData = []
    psfSize = np.linalg.norm(psf.shape)/2
    psfCentre = (psf.shape[0]//2, psf.shape[1]//2)

    for i in range(len(psf)):
        for j in range(len(psf[0])):
            if (i,j) != psfCentre and psf[i][j] > cutoff:
                xData.append(np.linalg.norm(np.asarray((i,j))-np.asarray(psfCentre))/psfSize)
                yData.append(psf[i][j])
    
    popt, pcov = curve_fit(funcPSF, xData, yData, p0=[100])

    print(popt,pcov)
    newPSF = np.zeros((len(psf),len(psf[0])))
    for i in range(len(newPSF)):
        for j in range(len(newPSF[0])):
            newPSF[i,j] = funcPSF(np.linalg.norm(np.asarray((i,j))-np.asarray((len(psf)//2,len(psf[0])//2)))/np.linalg.norm(np.asarray((len(psf)//2,len(psf[0])//2))),popt[0])

    newPSF[len(psf)//2,len(psf[0])//2] = 1
    return newPSF