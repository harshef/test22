from random import uniform
import numpy as np


def Repair_Solution( y, x):
    '''
    
    :param map_matrix: Mapped matrix
    :param y: Trial solution
    :param x: population
    :return: Repaired solution of y
    '''
    yRep = []

    return_max = []
    return_min = []

    print "Population inside repair solution : ",  x
    for k in range(len(y)):
        xmin = min(x[:,k])
        xmax = max(x[:,k])
        #print "XMAX for column {}  : ".format(k),xmax
        #print "XMIN for column {}  : ".format(k), xmin
        if y[k] < xmin:
            yRep.insert(k,xmin)
        elif y[k] > xmax:
            yRep.insert(k,xmax)
        else:
            yRep.insert(k,y[k])
        return_max.insert(k,xmax)
        return_min.insert(k, xmin)


    return yRep,return_max,return_min

def Select_Random(Q, population,i):
    '''
    :param mapped_matrix: Codebook Matrix
    :param population: Poulation
    :param Q: Mating Pool
    :return: Ranndom two parents from Q
    '''
    y1 = []
    y2 = []


    np.random.shuffle(Q)
    print "Mating pool inside solution generation for solution  {0}  : ".format(i),Q
    rand_neuron1=Q[0]
    rand_neuron2=Q[1]
    print "Mating Pool : ", Q
    print "random solution selected : ", rand_neuron1, rand_neuron2
    print "Population matrix inside Solution genration", population

    for j in range(len(population[0])):
        y1.insert(j, population[int(rand_neuron1)][j])
        y2.insert(j, population[int(rand_neuron2)][j])
    print "Random solution 1: ", y1
    print "Random solution 2: ", y2

    return y1, y2


def Mutation(yRval,di,xmax,xmin):
    '''

    :param yRval: Value at ith index of Repaired solution
    :param x: Current population
    :param di: Mutated value
    :return:
    '''
    nsol=(yRval + di * (xmax - xmin))

    return nsol
