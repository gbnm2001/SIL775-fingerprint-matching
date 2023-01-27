import math
from preprocessing import *
from fingerPrintOrientation import *
from geneticalgorithm import geneticalgorithm as ga
import math

imagecount = 0
def convert(map):
    M = []
    for (x,y) in map:
        M.append((x,y,map[(x,y)][1]))
    return M

#(img1, map1) = fingerprint_recognition("fingerprints/DB_1/102_2.tif")
#(img2, map2) = fingerprint_recognition("fingerprints/DB_1/104_5.tif")
#M1 = convert(map1)
#M2 = convert(map2)


def transform(tup, change):
    angle = math.radians(change[2])
    xn = change[3]*(tup[0]*cos(angle) - tup[1]*sin(angle)) + change[0]
    yn = change[3]*(tup[0]*sin(angle) + tup[1]*cos(angle)) + change[1]
    return (xn,yn)

def pairing(minutia1, minutia2, dist_thresh, theta_thresh, change, match_ratio=0.08):
    print('Transform values = ',change)
    m = len(minutia1)
    n = len(minutia2)
    f1 = [False for i in range(m)]
    f2 = [False for i in range(n)]
    matches = []
    count=0
    for i in range(m):
        for j in range(n):
            if ((not f2[j]) and (not f1[i])):
                (x1,y1) = transform(minutia1[i], change)
                t1 = minutia1[i][2]+change[2]
                if(dist((x1,y1), minutia2[j]) < dist_thresh and abs(t1-minutia2[j][2])<theta_thresh):
                    count+=1
                    print(x1,y1,minutia2[j][0], minutia2[j][1])
                    matches.append((i,j))
                    f1[i]=True
                    f2[j]=True
    print('matching points ', count)
    if(count > match_ratio*min(n,m)):#80% of template
        return True
    else:
        return False


################HOUGH TRANSFORM MARTCHING
def houghTransform(minutia1, minutia2):
    '''
    input - list of tuples (x,y,theta(degrees)) of two finger prints
    output - delta_x, delta_y, delta_theta (degrees)
    '''
    m = len(minutia1)
    n = len(minutia2)
    A = {}
    for i in range(n):
        for j in range(m):
            d_theta = radians(minutia2[i][2] - minutia1[j][2])
            d_x = round(minutia2[i][0] - minutia1[j][0]*cos(d_theta) - minutia1[j][1]*sin(d_theta))//5*5
            d_y = round(minutia2[i][1] + minutia1[j][0]*sin(d_theta) - minutia1[j][1]*cos(d_theta))//5*5
            d_theta = round(degrees(d_theta))//5*5
            if((d_x,d_y,d_theta) in A):
                A[(d_x,d_y,d_theta)] = A[(d_x,d_y,d_theta)] +1
            else:
                A[(d_x,d_y,d_theta)]=1
    M = max(A.values())
    #print(sorted(list(A.keys())))
    print('Max common value in hough pairing = ',M)
    opt = None
    for k in A:
        if(A[k] == M):
            return k



    



def houghMatching(path1, path2, dist_thresh, theta_thresh):
    print('Running hough matching')
    t1 = ThreadWithReturn(target=getMinutiae, args=(path1,))
    t1.start()
    t2 = ThreadWithReturn(target=getMinutiae, args=(path2,))
    t2.start()
    M1=t1.join()
    M2=t2.join()

    delta = houghTransform(M1, M2)
    delta = (delta[0],delta[1],delta[2],1)
    res = pairing(M1, M2, dist_thresh, theta_thresh, delta)
    print('HOUGH MATCH = ',res)
    return res




################################GENETIC ALGORITHM MATCHING
def costFunction(solution,M1,M2):
    n1 = len(M1)
    n2 = len(M2)
    total_dist_s = 0
    if(n1<n2):
        for j in range(n1):
            dj = float('inf')
            for k in range(n2):
                dj = min(dj, dist(transform(M1[j], solution), M2[k]))
            total_dist_s +=dj**2
        return total_dist_s
    else:
        for j in range(n2):
            dj = float('inf')
            for k in range(n1):
                dj = min(dj, dist(transform(M1[k], solution), M1[j]))
            total_dist_s +=dj**2
        return total_dist_s

def geneticAlogorithm(path1, path2, dt,tt):
    print('Running genetic algorithm')
    t1 = ThreadWithReturn(target=getMinutiae, args=(path1,))
    t1.start()
    t2 = ThreadWithReturn(target=getMinutiae, args=(path2,))
    t2.start()
    M1=t1.join()
    M2=t2.join()
    varbound=np.array([[-100,100],[-100,100],[-30,30],[0.9,1.1]])
    vartype=np.array([['int'],['int'],['int'],['real']])
    algorithm_param = {'max_num_iteration': 100,\
                    'population_size':100,\
                    'mutation_probability':0.1,\
                    'elit_ratio': 0.01,\
                    'crossover_probability': 0.5,\
                    'parents_portion': 0.7,\
                    'crossover_type':'uniform',\
                    'max_iteration_without_improv':None}
    model=ga(function= lambda x: costFunction(x, M1,M2),dimension=4,
    variable_type_mixed=vartype,variable_boundaries=varbound,
    algorithm_parameters=algorithm_param, convergence_curve=False)

    model.run()
    delta = model.output_dict
    print(model.output_dict)
    res = pairing(M1, M2, dt, tt, delta['variable'],0.01)
    print('GENETIC ALGORITHM MATCH = ', res)
    return res

####################CORE POINT MATCHING
def getSlopeOfBestFit(points):
    avg_x, avg_y = 0, 0
    for (x,y) in points:
        avg_x +=x
        avg_y+=y
    n = len(points)
    avg_y = avg_y/n
    avg_x = avg_x/n
    num, den = 0, 0
    for (x,y) in points:
        num += (x-avg_x)*(y-avg_y)
        den += (x-avg_x)**2
    angle = round(degrees(atan2(num,den)))
    return angle
    
def getCoreAngle(t1, block_size=10):
    global imagecount
    (nr, nc) = t1.shape
    t1 = t1[int(0.2*nr):int(0.8*nr), int(0.2*nc):int(0.8*nc)]
    (nr1, nc1) = t1.shape
    w = block_size
    variance_img = np.zeros(t1.shape)
    max_r1,max_c1, maxV1 = 0, 0, 0
    max_var_points = []#for each row get the column which has max variance
    for r in range(0,nr1//w):
        max_c, maxV = 0, 0
        for c in range(0,nc1//w):
            avg = 0
            V = np.var(t1[r*w:w*(r+1), c*w:(c+1)*w])
            variance_img[r*w][c*w] = V
            if(V> maxV and V>100):
                max_c = c*w
                maxV = V
            if(V>maxV1):
                max_r1 = r*w
                max_c1 = c*w
                maxV1 = V
        max_var_points.append((r*w,max_c))
    print('shape of matrix ', nr, nc)
    print('max variance points', max_var_points)
    res = (0.2*nr+max_r1+block_size//2,0.2*nc+max_c1+block_size//2, getSlopeOfBestFit(max_var_points))
    print(res)
    variance_img[max_r1][max_c1] = 128
    print('max variance ',variance_img[max_r1][max_c1])
    print('showing angle variance image')
    showArr(variance_img, f'variance image {imagecount}', [(max_r1, max_c1)])
    imagecount+=1
    return res
    


def corePointMatching(imagepath1, imagepath2, dist_thresh=10, theta_thresh=10):
    '''
    ALGORITHM
    FIND THE CORE POINT COORDINATES
        FIND COORDINATE S OF HIGHEST VARIANCE IN ORIENTATION ANGLES (CORE POINT)
        FIND COORDINATES OF HIGH VARIANCE IN ORIENTATION ANGLES IN EACH ROW
        PERFORM LINEAR REGRESSION ON THOSE POINTS TO GET THE ANGLE AXIS (THETA)
        RETURN (CORE POINTS, THETA)
    '''
    print('Running core point match')
    Ie1 = enhancement(normalization(segmentation(plt.imread(imagepath1))))[10:-10,10:-10]
    thetas1 = ridge_orientation_field(np.float64(numpyInvert(Ie1)))
    (cx1,cy1,ct1) = getCoreAngle(thetas1)
    showArr(Ie1,'Enhanced1', [(cx1, cy1)])
    
    
    
    Ie2 = enhancement(normalization(segmentation(plt.imread(imagepath2))))[10:-10,10:-10]
    thetas2 = ridge_orientation_field(np.float64(numpyInvert(Ie2)))
    (cx2,cy2,ct2) = getCoreAngle(thetas2)
    showArr(Ie2,'Enhanced2', [(cx2, cy2)])
    
    trans = (cx2-cx1, cy2-cy1, ct2-ct1, 1)#scale is kept one for now
    print(trans)
    '''
    binarization(Ie1)
    It1 = numpyInvert(thinning(Ie1))
    M1 = minutiaeExtraction(It1, thetas1,10)
    binarization(Ie2)
    It2 = numpyInvert(thinning(Ie2))
    M2 = minutiaeExtraction(It2, thetas2,10)
    #res = pairing(M1, M2, dist_thresh, theta_thresh, trans)
    #print('CORE POINT MATCH = ', res)
    '''


p1 = "fingerprints/DB1_B/105_7.tif"
p2 = "fingerprints/DB1_B/106_1.tif"

#houghMatching(p1, p2, 20,20)

geneticAlogorithm(p1, p2, 20,20)

#corePointMatching(p1,p2)