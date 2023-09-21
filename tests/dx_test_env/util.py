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