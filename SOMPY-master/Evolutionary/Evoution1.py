import math
import operator
from random import uniform
import numpy as np
from collections import defaultdict
import numpy as np
import primefac
from DifferentialEvolution.differential_evolution_functions import Mutation,Repair_Solution,Select_Random
from collections import defaultdict
data=np.array
matrix=np.array

def mapping(data, matrix):
    """

    :param data: Original Normalised data to be mapped
    :param matrix: Neuron matrix on which "data" is to be mapped
    :return: Mapped matrix "matrix"
    """
    dd={} #Dictionary for distance
    print "current codebook Matrix :    ", matrix,
    print "Current population :  ", data
    print "MAtrix size",matrix.size, data.size
    if matrix.size==data.size:
        U = np.arange(data.shape[0]) #Returns no. of neuron in lattice
        print "Total neurons in lattice",U
        for i in range(len(data)): # For each poopulation neuron
            temp1 = np.copy(data[i])
            for j in range(len(matrix)):
                temp2 = np.copy(matrix[j])
                dist = np.linalg.norm(temp1 - temp2)
                dd[j]=dist

            sorted_x = sorted(dd.items(), key=operator.itemgetter(1)) #Sort the dictionary in ascending order
            print "Sorted Dict",sorted_x
            print "Current chromosome no-",i
            print "current Neurons no.   :  ",U
            for k,v in sorted_x:
                found=0
                for m in range((len(U))):
                    if k == U[m]:
                        matrix[k]=temp1
                        found_index = np.where(U == k)
                        print "Found Index:   ",found_index
                        U=np.delete(U,found_index)
                        found=1
                        print "Updated neurons left : ",U
                        break
                if found==0:
                    continue
                else:
                    break
    print "Updated codebook Matrix  after one to one mapping :    ", matrix,

    return matrix

def generate_matingPool(H,i, matrix,beta=0):
    """

    :param H: Number of neurons in mating pool of each neuron (size of H<=no. of neuron in Lattice)
    :param matrix: initial Population
    :return: Mating pool matrix where each row represents a pool of neuron of neuron represented by the row index
    """
    dd={}
    print "Codebook matrix for mating pool generation: \n  ", matrix
    mating_pool=[]
    flag=0
    if uniform(0,1)<beta:
        flag=0
        temp1=matrix[i]
        for j in range(len(matrix)):
            temp2 = matrix[j]
            dist = np.linalg.norm(temp1 - temp2)
            dd[j] = dist
        sorted_x = sorted(dd.items(), key=operator.itemgetter(1))
        print "Sorted distance for neuron {0} to other neurons :  \n".format(i),sorted_x
        counter=0
        for k, v in sorted_x:
            if counter<H:
                mating_pool.insert(counter,k)
                counter+=1
    else:
        flag=1
        print "Random probability is greater than Beta for this neuron therefore assign population as the mating pool for neuron {0}".format(i)
        mating_pool=list(np.arange(0, len(matrix)))
        print  "Mating pool for neuron {}".format(i),mating_pool
    return np.asarray(mating_pool),flag

def Generate(Q,  H, population,i,CR=0,F=0, mutation_prob=0,eta=0):
    """

    :param Q: Mating pool
    :param mapp: Mapped matrix
    :param H: Size of mating pool
    :param population: Population
    :return: New solution
    
    """
    #print "-------solution generation-----------\n"
    print "mating pool  inside generate function :   ", Q
    yRep = []
    y1, y2=Select_Random(Q,population,i)
    y_dash = []
    x=population[i]
    nsol = np.zeros((len(x)))
    print "Random selected ",y1,y2
    for j in range(len(y1)):
        if CR >= uniform(0,1): # CR value specified as 0.8
            m = y1[j] - y2[j]
            y_dash.insert(j, x[j] + F * m)  # Value of taken as F=0.9
        else:
            y_dash.insert(j, x[j]) #Otherwise
    y_dash=np.array(y_dash)
    yRep,xmax,xmin = Repair_Solution(y_dash, population)  # Repaired solution

    for m in range(len(yRep)):
        if mutation_prob > uniform(0,1):  # Mutation probability
            DeltaR = uniform(0, 1)
            if DeltaR < 0.5:
                di = (math.pow(math.pow(((2 * DeltaR) + (1 - 2 * DeltaR) * ((xmax[m] - yRep[m]) / (xmax[m] - xmin[m]))), eta + 1), 1 / (eta + 1)))-1
            else:
                di = 1-(math.pow(math.pow(2 - 2 * DeltaR + (2 * DeltaR - 1) * ((yRep[m] - xmin[m]) / xmax[m] - xmin[m]), eta + 1), 1 / (eta + 1)))
            nsol[m]=Mutation(yRep[m], di,xmax[m],xmin[m])
        else:
            nsol[m] = yRep[m]
    print "New Solution", nsol

    return nsol


def Mapsize(n):

    factors = list( primefac.primefac(n) )
    print factors
    if len(factors)==1:
        return 1, n
    elif len(factors)==2:
        return factors[0],factors[1]
    else:
        x=int(math.floor(len(factors)/2))
        print x

        p1=1
        p2=1
        p3=1
        for i in range(x+1):
            p1=p1*factors[i]
        for j in range(x+1,len(factors)):
            p2 = p2 * factors[j]
        return p1,p2

def calculate_dunn_index( data, label,n_cluster=0):
    data=data.tolist()
    d1=[]
    d2=[]
    d3=[]
    cluster_data=defaultdict(list)
    for i in range (n_cluster):
        for j in range (len(label)):
            if label[j]==i:
                cluster_data[i].append(data[j])

    for k,v in cluster_data.iteritems():

        cluster_list=cluster_data[k]
        for i in range (len(cluster_list)):
            temp1=cluster_list[i]
            for j in range(len(cluster_list)):
                temp2 = cluster_list[j]

                dist = np.linalg.norm(np.asarray(temp1) - np.asarray(temp2))
                d1.insert(j,dist)

            d2.insert(i,max(d1))
            d1=[]

        d3.insert(k,max(d2))
        d2=[]
    xmax= max(d3)
    #################################################################################################
    #Calcuation of minimum distance
    d1=[]
    d2=[]
    d3=[]
    d4=[]
    for k, v in cluster_data.iteritems():
        cluster_list=cluster_data[k]

        for j in range(len(cluster_list)):
            temp1=cluster_list[j]
            for key,value in cluster_data.iteritems():
                if not key==k:
                    other_cluster_list=cluster_data[key]
                    for index in range ((len(other_cluster_list))):
                        temp2=other_cluster_list[index]
                        dist=np.linalg.norm(np.asarray(temp1)-np.asarray(temp2))
                        d1.insert(index,dist)

                    d2.insert(key,min(d1))
                    d1=[]
            d3.insert(j,min(d2))
            d2=[]
        d4.insert(k,min(d3))
    xmin=min(d4)
    dunn_index= xmin/xmax
    return dunn_index


def get_agglomerative_centers(label,Idata):
    cluster_dict=defaultdict(list)
    print "Idata",Idata
    for i in range(len(label)):
        for j in range(max(label)+1):
            if label[i]==j:
                cluster_dict[j].append(Idata[i])
                break
    print label
    print cluster_dict
    centers=[]
    for k in cluster_dict:
        print len(cluster_dict[k])
        val = cluster_dict.get(k)
        centers.insert(k , np.mean(np.asanyarray(val),axis=0))
    return_centers=[]
    for k in range(len(centers)):
        value=list(centers[k])
        return_centers.insert(k,value)
        k+=1
    print centers
    print return_centers
    return return_centers