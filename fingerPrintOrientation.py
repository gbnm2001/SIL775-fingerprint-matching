from preprocessing import *
import cv2
from math import sin,cos,atan2

I = plt.imread("fingerprints/DB_1/105_3.tif")
Ip = getPreprocessedImage(I)
np.set_printoptions(threshold=np.inf)

def plotField(theta_arr, block_size=10):
    (nr,nc) = theta_arr.shape
    out = np.zeros(theta_arr.shape)
    for r in range(0,nr,block_size):
        for c in range(0,nc,block_size):
            theta = theta_arr[r][c]
            #take the middle point of the block and extend in either directions
            #find the two extreme points
            (x1,y1) = (int(r+block_size/2*(1+cos(theta))), int(c + block_size/2*(1+sin(theta))) )
            (x2,y2) = (int(r+block_size/2*(1-cos(theta))), int(c + block_size/2*(1-sin(theta))) )
            x2 = max(0,min(x2,nr-1))
            y2 = max(0,min(y2,nc-1))
            x1 = max(0,min(x1,nr-1))
            y1 = max(0,min(y1,nc-1))
            out[x1][y1] = 255
            out[x2][y2] = 255
            out[(x1+x2)//2][(y1+y2)//2] = 255
            out[int(0.25*x1+0.75*x2)][int(0.25*y1+0.75*y2)] = 255
            out[int(0.25*x2+0.75*x1)][int(0.25*y2+0.75*y1)] = 255
    showArr(out)
    return

def ridge_orientation_field(normalized_image,block_size=10):
    scale,delta=1,0
    grad_x = cv2.Sobel(normalized_image, cv2.CV_64F, 0, 1, ksize=3,scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(normalized_image, cv2.CV_64F, 1, 0, ksize=3,scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    thetas = np.zeros(normalized_image.shape)
    (nr,nc) = normalized_image.shape
    for r in range(0,nr, block_size):
        for c in range(0,nc, block_size):
            numer = 0
            denom = 0
            r1=min(r+block_size, nr)
            c1 = min(c+block_size, nc)
            for i in range(r,r1):
                for j in range(c, c1):
                    numer+=2*grad_x[i][j]*grad_y[i][j]
                    denom += (grad_x[i][j]**2 - grad_y[i][j]**2)
            thetas[r][c] = 0.5*atan2(numer,denom)+ np.pi/2
    #print(thetas)
    return thetas

In = enhancement( normalization(segmentation(plt.imread("fingerprints/DB_1/101_3.tif"))))
showArr(In)
a = ridge_orientation_field(In)
plotField(a)


