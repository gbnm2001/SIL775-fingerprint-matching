import math
from preprocessing import *
from fingerPrintOrientation import *
from geneticalgorithm import geneticalgorithm as ga
import math
def houghTransform(minutia1, minutia2):
    '''
    input - list of tuples (x,y,theta) of two finger prints
    output - delta_x, delta_y, delta_theta
    '''

    m = len(minutia1)
    n = len(minutia2)
    A = {}
    for i in range(n):
        for j in range(m):
            d_theta = round(minutia2[i][2] - minutia1[j][2],1)
            d_x = round(minutia2[i][0] - minutia1[j][0]*math.cos(d_theta) - minutia1[j][1]*math.sin(d_theta))//5*5
            d_y = round(minutia2[i][1] + minutia1[j][0]*math.sin(d_theta) - minutia1[j][1]*math.cos(d_theta))//5*5
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

def corePointMatching(M1,M2):
    '''
    ALGORITHM
    FIND THE CORE POINT COORDINATES
        FIND COORDINATE S OF HIGHEST VARIANCE IN ORIENTATION ANGLES (CORE POINT)
        FIND COORDINATES OF HIGH VARIANCE IN ORIENTATION ANGLES IN EACH ROW
        PERFORM LINEAR REGRESSION ON THOSE POINTS TO GET THE ANGLE AXIS (THETA)
        RETURN (CORE POINTS, THETA)
    '''

def transform(tup, change):
    angle = math.radians(change[2])
    xn = change[3]*(tup[0]*cos(angle) - tup[1]*sin(angle)) + change[0]
    yn = change[3]*(tup[0]*sin(angle) + tup[1]*cos(angle)) + change[1]
    return (xn,yn)

def pairing(minutia1, minutia2, dist_thresh, theta_thresh, change, match_ratio=0.7):
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
                t1 = minutia1[i][2]+math.radians(change[2])
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

def houghMatching(M1, M2, dist_thresh, theta_thresh):
    delta = houghTransform(M1, M2)
    delta = (delta[0],delta[1],math.degrees(delta[2]),1)
    res = pairing(M1, M2, dist_thresh, theta_thresh, delta)
    print('HOUGH MATCH = ',res)
    return res

# M1 = getMinutiae("fingerprints/DB_1/101_3.tif")
# M2 = getMinutiae("fingerprints/DB_1/101_3.tif")

#print(houghPairing(M1,M2,60,0.5))


def fitnessFunction(solution):
    n1 = len(M1)
    n2 = len(M2)
    total_dist_s = 0
    for j in range(n1):
        dj = float('inf')
        for k in range(n2):
            dj = min(dj, dist(transform(M1[j], solution), M2[k]))
        total_dist_s +=dj**2
    return total_dist_s

def geneticAlogorithm(M1, M2, dt,tt):
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
    model=ga(function=fitnessFunction,dimension=4,
    variable_type_mixed=vartype,variable_boundaries=varbound,
    algorithm_parameters=algorithm_param, convergence_curve=False)

    model.run()
    delta = model.output_dict
    print(model.output_dict)
    res = pairing(M1, M2, dt, tt, delta['variable'],0.5)
    print('GENETIC ALGORITHM MATCH = ', res)
    return res

#houghMatching(M1, M2, 30,1)
t1 = ThreadWithReturn(target=getMinutiae, args=("fingerprints/DB_1/103_3.tif",))
t1.start()

t2 = ThreadWithReturn(target=getMinutiae, args=("fingerprints/DB_1/103_5.tif",))
t2.start()

M1=t1.join()
M2=t2.join()
geneticAlogorithm(M1, M2, 25,0.2)