import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
import fingerprint_enhancer
from skimage import morphology

def getMean(image_arr):
    pass

def getVariance(image_arr,):
    pass

#1
def segmentation(image_arr, block_size = 10):
    '''
    CALCULATE variance in w*w windows
    delete the border blocks with variance less than threshold (0.1*total_variance)
    output is a 2D cropped array
    '''
    (nr,nc) = image_arr.shape
    left = nc
    right = 0
    bottom = nr
    top = 0
    w = block_size
    #get total vairance
    varTh = np.var(image_arr)*0.1

    #iterate w*w blocks
    for r in range(0,nr//w):
        for c in range(0,nc//w):
            #caclulate mean in that region
            avg = 0
            for r1 in range(r*w, (r+1)*w):
                for c1 in range(c*w, (c+1)*w):
                    avg += image_arr[r1][c1]
            avg = avg/(w**2)
            #calculate the variance
            V=0
            for r1 in range(r*w, (r+1)*w):
                for c1 in range(c*w, (c+1)*w):
                    V += (image_arr[r1][c1]-avg)**2
            V = V/(w**2)
            if(V>=varTh):
                left = min(left, c*w)
                right = max(right, (c+1)*w)
                top = max(top, (r+1)*w)
                bottom = min(bottom, r*w)
    
    return image_arr[bottom:top+1, left:right+1]

#2
def normalization(image_arr, M0=128, V0=10000):
    '''
    NORMALIZE THE VALUES TO EXPECTED MEAN AND VARIANCE
    OPTIONAL I THINK
    mean=128
    variance=128
    this seems unnecessary
    '''
    (nr,nc) = image_arr.shape
    M = np.mean(image_arr)
    V = np.var(image_arr)
    def helper(x):
        if (x>M):
            return M0 + (V0*(x - M)**2/V)**0.5
        else:
            return M0 - (V0*(x - M)**2/V)**0.5
    numpyHelper = np.vectorize(helper)
    new_image = numpyHelper(image_arr)
    return new_image

#3
def enhancement(image_arr):
    '''
    RIDGE JOINING
    GABOR filter
    opencv has gabor filter
    theta = pi/4
    lambda = pi/4
    ksize = TBD
    sigma = 5
    gamma = 0.9
    phi = 0
    '''
    return fingerprint_enhancer.enhance_Fingerprint(image_arr)

#4
def binarization(image_arr):
    '''
    CONVERT GRAYSCALE TO BINARY IMAGE
    requires Otsu's thresholding
    '''
    (nr,nc) = image_arr.shape
    for r in range(nr):
        for c in range(nc):
            if(image_arr[r][c] >128):
                image_arr[r][c] = 255
            else:
                image_arr[r][c] = 0
    return image_arr

def thinning(image_arr):
    '''
    REDUCE THE THICKNESS OF RIDGES TO ONE PIXEL
    CAN USE cv2 or maholas depending on the correctness
    '''

    thinned =  morphology.thin(image_arr)#np.where(skeletonize(image_arr//255), 0.0, 1.0)
    return thinned

def invert(x):
    if(x):
        return 0
    else:
        return 255

numpyInvert = np.vectorize(invert)


def showArr(image_arr):
    image = im.fromarray(image_arr)
    image.show()

def getPreprocessedImage(image_arr):
    #I = plt.imread("fingerprints/DB_1/105_3.tif")
    Is = segmentation(image_arr,10)
    In = normalization(Is, 200, 10000)
    Ie = enhancement(In)
    binarization(Ie)
    It = numpyInvert(thinning(Ie))
    return It

def saveImage(image_arr):
    np.set_printoptions(threshold=np.inf)


def minutiaeExtraction(thin_image):
    (nr,nc) = thin_image.shape
    ridgeEndings = []#list of tuples
    bifurcationPts = []
    for r in range(nr-2):
        for c in range(nc-2):
            #is end - only two dark pixels
            count = 0
            for i in range(3):
                for j in range(3):
                    if(thin_image[r+i][c+j] == 0):
                        count+=1

            if(thin_image[r+1][c+1] == 0 and count == 2):
                ridgeEndings.append((r+1,c+1))
            elif(thin_image[r+1][c+1]==0 and count==4):
                bifurcationPts.append((r+1,c+1))
    minutiae = ridgeEndings+bifurcationPts
    return minutiae

def minutiaRemoval(thin_image, ends, bifurs):
    '''
    REMOVE CLUSTERS
    REMOVE MINUTIAE NEAR BORDER
    RIDGE ENDINGS WITHIN D1 FACING EACH OTHER ARE REMOVED, this by itself should remove clusters
    RIDGE ENDING AND BIFURCATION WITHIN D2 ARE REMOVED
    TWO BIFURCATION WITHIN D3 ARE REMOVED
    '''
    B = 10
    D1 = 6
    D2 = 10
    D3 = 10
    (nr,nc) = thin_image.shape
    endDel = []
    for i in range(len(ends)):
        if(ends[i][0]<B or nr-ends[i][0]<B or ends[i][1]<B or nc-ends[i][1]<B):
            endDel.append(i)
    bifurDel = []
    endDel.reverse()
    for e in endDel:
        del ends[e]

    for i in range(len(bifurs)):
        if(bifurs[i][0]<10 or nc-bifurs[i][0]<10 or bifurs[i][1]<B or nc-bifurs[i][1]<B):
            bifurDel.append(i)
    bifurDel.reverse()
    for b in bifurDel:
        del bifurs[b]
    
    

    while(True):
        #find two near minutia
        mn = len(ends)
        deli=-1
        delj=-1
        for i in range(mn):
            for j in range(mn):
                if(i!=j and ((ends[i][0]-ends[j][0])**2 + (ends[i][1]-ends[j][1])**2)**0.5<D1):
                    deli=i
                    delj=j
                    break
            if(deli !=-1):
                break
        if(deli!=-1 and delj!=-1):
            del ends[max(deli,delj)]
            del ends[min(deli,delj)]
        else:
            break
    
    while(True):
        #find two near minutia
        en = len(ends)
        bn = len(bifurs)
        deli=-1
        delj=-1
        for i in range(en):
            for j in range(bn):
                if(i!=j and ((ends[i][0]-bifurs[j][0])**2 + (ends[i][1]-bifurs[j][1])**2)**0.5<D2):
                    deli=i
                    delj=j
                    break
            if(deli !=-1):
                break
        if(deli!=-1 and delj!=-1):
            del ends[max(deli,delj)]
            del bifurs[min(deli,delj)]
        else:
            break

    while(True):
        mn = len(bifurs)
        deli=-1
        delj=-1
        for i in range(mn):
            for j in range(mn):
                if(i!=j and ((bifurs[i][0]-bifurs[j][0])**2 + (bifurs[i][1]-bifurs[j][1])**2)**0.5<D3):
                    deli=i
                    delj=j
                    break
            if(deli !=-1):
                break
        if(deli!=-1 and delj!=-1):
            del bifurs[max(deli,delj)]
            del bifurs[min(deli,delj)]
        else:
            break
                    
    pass


    

#showArr(getPreprocessedImage(plt.imread("fingerprints/DB_1/101_3.tif")))