import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from PIL import ImageDraw
import fingerprint_enhancer
from skimage import morphology



#1
def segmentation(image_arr, block_size = 5):
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
    varTh = np.var(image_arr)*0.3

    #iterate w*w blocks
    for r in range(0,nr//w):
        for c in range(0,nc//w):
            V = np.var(image_arr[r*w:(r+1)*w,c*w:(c+1)*w])#V/(w**2)
            if(V>=varTh):
                left = min(left, c*w)
                right = max(right, (c+1)*w)
                top = max(top, (r+1)*w)
                bottom = min(bottom, r*w)
    
    return image_arr[bottom:top, left:right]

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
    '''
    return fingerprint_enhancer.enhance_Fingerprint(image_arr)

#4
def binarization(image_arr):
    '''
    CONVERT GRAYSCALE TO BINARY IMAGE
    requires Otsu's thresholding
    '''
    _,image_arr = cv2.threshold(image_arr,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return image_arr

def thinning(image_arr):
    '''
    REDUCE THE THICKNESS OF RIDGES TO ONE PIXEL
    '''
    thinned =  morphology.thin(image_arr)#np.where(skeletonize(image_arr//255), 0.0, 1.0)
    return thinned

def invert(x):
    if(x):
        return 0
    else:
        return 255

numpyInvert = np.vectorize(invert)


def showArr(image_arr, name, points = []):
    image = im.fromarray(image_arr)
    rgbimg = im.new("RGBA", image.size)
    rgbimg.paste(image)
    draw = ImageDraw.Draw(rgbimg)
    for point in points:
        draw.rectangle((point[1],point[0],point[1]+5,point[0]+5), fill= (255,0,0,128))
    rgbimg.show(title=name)
    

def getPreprocessedImage(image_arr):
    #I = plt.imread("fingerprints/DB_1/105_3.tif")
    Is = segmentation(image_arr,10)
    In = normalization(Is, 200, 10000)
    Ie = enhancement(In)
    binarization(Ie)
    It = numpyInvert(thinning(Ie))
    return It

def saveImage(image_arr):
    f = open('arr.txt','w+')
    f.write(str(image_arr))
    f.close()


def minutiaeExtraction(thin_image, thetas, block_size=10):
    (nr,nc) = thin_image.shape
    ridgeEndings = []#list of tuples
    bifurcationPts = []
    for r in range(nr-2):
        for c in range(nc-2):
            #is end - only two dark pixels
            count = 0
            for i in range(3):
                for j in range(3):
                    if(thin_image[r+i][c+j] < 10):
                        count+=1
            theta = thetas[r][c]
            if(thin_image[r+1][c+1]==0 and count == 2):
                ridgeEndings.append((r+1,c+1,theta))
            elif(thin_image[r+1][c+1]==0 and count>=4):
                bifurcationPts.append((r+1,c+1,theta))

    minutiaRemoval(thin_image,ridgeEndings, bifurcationPts)
    return ridgeEndings+bifurcationPts

def boundaryRemoval(ends, bifurs, shape, B=15):
    (nr,nc) = shape
    endDel = []
    bifurDel = []
    for i in range(len(ends)):
        if(ends[i][0]<B or nr-ends[i][0]<B or ends[i][1]<B or nc-ends[i][1]<B):
            endDel.append(i)
    bifurDel = []
    endDel.reverse()
    for e in endDel:
        del ends[e]

    for i in range(len(bifurs)):
        if(bifurs[i][0]<B or nr-bifurs[i][0]<B or bifurs[i][1]<B or nc-bifurs[i][1]<B):
            bifurDel.append(i)
    bifurDel.reverse()
    for b in bifurDel:
        del bifurs[b]

def dist(t1,t2):
    return ((t1[0]-t2[0])**2 + (t1[1]-t2[1])**2)**0.5

def nearEndPtsRemoval(ends,D1=6):
    while(True):
        #find two near endingPoints
        mn = len(ends)
        deli=-1
        delj=-1
        for i in range(mn):
            for j in range(mn):
                if(i!=j and dist(ends[i],ends[j])<D1):
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

def nearEndBifurRemoval(ends, bifurs, D2 = 10):
        while(True):
            #find two near minutia
            en = len(ends)
            bn = len(bifurs)
            deli=-1
            delj=-1
            for i in range(en):
                for j in range(bn):
                    if(i!=j and dist(ends[i],bifurs[j])<D2):
                        deli=i
                        delj=j
                        break
                if(deli !=-1):
                    break
            if(deli!=-1 and delj!=-1):
                del ends[deli]
                del bifurs[delj]
            else:
                break

def nearBifurRemoval(bifurs, D3=10):
        while(True):
            mn = len(bifurs)
            deli=-1
            delj=-1
            for i in range(mn):
                for j in range(mn):
                    if(i!=j and dist(bifurs[i],bifurs[j])<D3):
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

def removeCluster(ends, bifurs,nr,nc,D0=10):
    clusters = {}
    minutiae = [(i,True) for i in ends] + [(i,False) for i in bifurs]
    mn = len(minutiae)
    count = 0
    #GROUP clusters
    while(len(minutiae)>0):
        clusters[count] = [minutiae[0]]
        del_list = []
        tn = len(minutiae)
        for i in range(1,tn):
            if(dist(minutiae[0][0],minutiae[i][0]) < D0):
                del_list.append(i)
        del_list.reverse()
        for j in del_list:
            clusters[count].append(minutiae[j])
            del minutiae[j]
        del minutiae[0]
        count+=1
    
    #SORT the clusters and retain only the central one
    ends.clear()
    bifurs.clear()
    for i in clusters:
        clusters[i].sort(key = lambda x: x[0][0]*nc+x[0][1])
        mid = clusters[i][len(clusters[i])//2]
        if(mid[1]):
            ends.append(mid[0])
        else:
            bifurs.append(mid[0])
    
    return

def minutiaRemoval(thin_image, ends, bifurs):
    '''
    REMOVE CLUSTERS
    REMOVE MINUTIAE NEAR BORDER
    RIDGE ENDINGS WITHIN D1 FACING EACH OTHER ARE REMOVED, this by itself should remove clusters
    RIDGE ENDING AND BIFURCATION WITHIN D2 ARE REMOVED
    TWO BIFURCATION WITHIN D3 ARE REMOVED
    '''
    B = 15
    D0=10
    D1 = 6
    D2 = 10
    D3 = 10
    (nr,nc) = thin_image.shape
    boundaryRemoval(ends,bifurs,thin_image.shape,B)
    removeCluster(ends,bifurs, thin_image.shape[0], thin_image.shape[1], D0)
    nearEndPtsRemoval(ends, D1)
    nearEndBifurRemoval(ends, bifurs, D2)
    nearBifurRemoval(bifurs,D3)
    print('no. ends = ',len(ends),'no. bifurcations = ', len(bifurs))
    
    return


    
