import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

def GetSourcePts(mask, filePath=None):
    """
    Get the source points from the mask and save them to a file
    :param mask: mask to get the source points from
    :param filePath: path to save the source points to
    :return: the source points
    """
    sourcePts = []

    (maskWidth, maskHeight) = mask.shape

    for i in range(maskWidth):
        for j in range(maskHeight):
            if mask[i,j] != 0:
                sourcePts.append([i,j])
    if filePath is not None:
        with open(filePath + '/sourcePts.pkl', 'wb') as file: 
            pickle.dump(sourcePts, file)

    return sourcePts


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

def EuclideanDistance(x,y):
    """
    Find the Euclidean distance between two points x and y
    :param x: first point
    :param y: second point
    :return: Euclidean distance between x and y
    """
    return ((x[0]-y[0])**2 + (x[1]-y[1])**2)**0.5

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
    return model.getPSF(torch.tensor([[EuclideanDistance(origin,target)/diagonal]])).detach().numpy()[0]
    
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
    
def TrainingLoop(model,dataloader,lossCriterion,optimizer,epochs=50,savePath=None,displayGraph=False):
    """
    This is our main training loop for our PSF Net
    :param model: model to be trained
    :param dataloader: dataloader to be used for training
    :param lossCriterion: loss criterion to be used for training
    :param optimizer: optimizer to be used for training
    :param epoch: number of epochs to train for
    :param savePath: path to save the model to
    :param displayGraph: whether to display the graph of the psf
    :return: None
    """
    
    running_loss = 0.0
    for e in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            inputs, outputs = data
            optimizer.zero_grad()
            nn_outputs = model(inputs)
            loss = lossCriterion(nn_outputs.float(), outputs.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (e%20==0):
                print("Epoch: ",e+1, " Loss: ",running_loss)
                if (displayGraph):
                    #Normalise x axis of original psf to 1
                    psf_x = [np.linspace(0,1,100,dtype=np.float32)]
                    psf_y = model.getPSF(torch.tensor(psf_x))
                    psf_x = psf_x[0]
                    psf_y = psf_y.detach().numpy().flatten()
                    plt.plot(psf_x,psf_y)
                    plt.show()
                    
    print("Epoch: ",epochs, " Loss: ",running_loss)
    if savePath is not None:
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss,
            }, savePath)
        
def InverseMatrix(originalOutput, mask, sourcePts, PSFNet, learningRate = 0.5, randomPts = 0, adjPts = 0):
    """
    Performs Matrix Inversion to get the originalOutput image with the source pts contribution.
    sourcePts with the same mask value are forced to have the same intensity
    Returns a new image based on the originalOuput*(1-learningRate) + learningRate*newOutput

    :param originalOutput: original output image
    :param mask: mask of the original output image
    :param sourcePts: source points of the original output image
    :param PSFNet: our trained model
    :param learningRate: learning rate of the matrix inversion
    :param randomPts: number of random points to be included in the matrix inversion
    :return: new image based on the originalOuput*(1-learningRate) + learningRate*newOutput
    """
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

    #Get adjacent points of source points
    for (x,y) in sourcePts:
        for i in range(-adjPts,adjPts+1):
            for j in range(-adjPts,adjPts+1):
                if x+i >= 0 and x+i < len(originalOutput) and y+j >= 0 and y+j < len(originalOutput[0]):
                    additionalPts.append([x+i,y+j])

    #Count number of unique masked points
    maskedPts = []
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i,j] not in maskedPts and mask[i,j] != 0:
                maskedPts.append(mask[i,j])

    originalOutputSize = EuclideanDistance([0,0],[len(originalOutput),len(originalOutput[0])])
    print(maskedPts)

    #Generate the matrix A
    A = np.zeros((len(sourcePts) + len(additionalPts), len(maskedPts)))
    for i, x in enumerate(additionalPts):
        for j, y in enumerate(sourcePts):
            A[i, maskedPts.index(mask[y[0]][y[1]])] += GetPSFModel(y,x,PSFNet,originalOutputSize)

    #Generate the vector b
    b = np.zeros((len(sourcePts) + len(additionalPts), 1))
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
