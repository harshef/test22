from sklearn.metrics.cluster import v_measure_score
import numpy as np
from numpy import genfromtxt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster.supervised import adjusted_rand_score
from sklearn.preprocessing import normalize
#from plotter import plot_max_Sil_Score
import sompy
from Evolutionary.Evoution1 import Mapsize, generate_matingPool,Generate,mapping,calculate_dunn_index
from nsga2.main import Select
from normalization import NormalizatorFactory
from sklearn import cluster
from sklearn.metrics import normalized_mutual_info_score
from  pbm_Index import cal_pbm_index
from visualization.umatrix import UMatrixView
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from visualization.hitmap import HitMapView
#from visualization.histogram import Hist2d

Iflag=0       #Flag
x = []        #store all center
DunnIndex=[]        #List of ari against each solution in population
sil_sco=[]    #List of Silhoutte Score angainst each solution in population
counter=0
actual_label=[]
TDS=[0]

data = genfromtxt('/home/acer/PycharmProjects/SOMBatch_FIX/SOMPY-master/data_set/spherical62.csv', delimiter='\t',usecols=(3))
for i in range(len(data)):
    actual_label.  insert(i,data[i])
Idata = genfromtxt('/home/acer/PycharmProjects/SOMBatch_FIX/SOMPY-master/data_set/spherical62.csv', delimiter='\t',usecols=(1,2))
#Idata = np.delete(Idata, 03 axis=0)
Initial_Label=[]
normalizer = NormalizatorFactory.build('var')
Idata=normalizer.normalize(Idata)
chromosome = int(input("Enter the no. of Chromosomes:"))
n = 6#int(input("Enter the no. of clusters to be formed:"))
Initial_Label=[]

"""
agglomerative= cluster.AgglomerativeClustering(n_clusters=n, linkage='average', affinity='euclidean')
idx= agglomerative.fit_predict(Idata)
hlabels=agglomerative.labels_
ss = silhouette_score(Idata, hlabels)
print "Silhoutte Score Agglomerative-",ss
di = calculate_dunn_index(Idata,hlabels,n_cluster=n)
print "Dunn Index Agglomerative",di
vms= v_measure_score(actual_label, hlabels)
print "V- measure score, Agglomerative",vms
nmi=normalized_mutual_info_score(actual_label,hlabels)
print "NMIS Agglomerative{}".format(nmi)
new_ari = adjusted_rand_score(actual_label, hlabels)
print "Agglomerative ARI=",new_ari
print "\n\n"

agglomerative= cluster.AgglomerativeClustering(n_clusters=n, linkage='ward', affinity='euclidean')
idx= agglomerative.fit_predict(Idata)
hlabels=agglomerative.labels_
ss = silhouette_score(Idata, hlabels)
print "Silhoutte Score Agglomerative-",ss
di = calculate_dunn_index(Idata,hlabels,n_cluster=n)
print "Dunn Index Agglomerative",di
vms= v_measure_score(actual_label, hlabels)
print "V- measure score, Agglomerative",vms
nmi=normalized_mutual_info_score(actual_label,hlabels)
print "NMIS Agglomerative{}".format(nmi)
new_ari = adjusted_rand_score(actual_label, hlabels)
print "Agglomerative ARI=",new_ari
print "\n\n"

agglomerative= cluster.AgglomerativeClustering(n_clusters=n, linkage='complete', affinity='euclidean')
idx= agglomerative.fit_predict(Idata)
hlabels=agglomerative.labels_
ss = silhouette_score(Idata, hlabels)
print "Silhoutte Score Agglomerative-",ss
di = calculate_dunn_index(Idata,hlabels,n_cluster=n)
print "Dunn Index Agglomerative",di
vms= v_measure_score(actual_label, hlabels)
print "V- measure score, Agglomerative",vms
nmi=normalized_mutual_info_score(actual_label,hlabels)
print "NMIS Agglomerative{}".format(nmi)
new_ari = adjusted_rand_score(actual_label, hlabels)
print "Agglomerative ARI=",new_ari
print "\n\n"

cluster = KMeans(n_clusters=n)
cluster.fit(Idata)
label = cluster.predict(Idata)
#print label
ss = silhouette_score(Idata, label)
print "Silhoutte Score K-means-",ss
di = calculate_dunn_index(Idata,label,n_cluster=n)
print "Dunn Index K-means",di
vms= v_measure_score(actual_label, label)
print "V- measure score, K-means",vms
nmi=normalized_mutual_info_score(actual_label,label)
print "NMIS Kmeans{}".format(nmi)
new_ari = adjusted_rand_score(actual_label, label)
print "ARI=",new_ari
"""
for i in range(1,chromosome+1):
    counter+=1
    pop = []
    cluster = KMeans(n_clusters=n,init='random',max_iter=1)
    cluster.fit(Idata)
    label = cluster.predict(Idata)
    centers = cluster.cluster_centers_
    print Idata.shape
    print "centers : \n", centers
    print "one center, length : ", len(centers[0])
    print "no center, ", len(centers), centers.shape
    print "Label : ", len(label)
    a=centers.tolist()
    print (a), len(a), len(a[0])
    print type(a), type(a[0])
    for j in range(len(a)):
        for k in range(len(Idata[0])):
            pop.append(a[j][k])
    print pop
    break
    x.insert(i,pop)
    ss=silhouette_score(Idata, label)
    pbm= cal_pbm_index(n,Idata,centers,label)
    #di = calculate_dunn_index(Idata,label,n_cluster=n)
    sil_sco.insert(i, ss)
    DunnIndex.insert(i,pbm)
    Initial_Label.append(label.tolist())
Final_label= list(Initial_Label)

population=np.array(x)
print "Population : ", population
print "PBM : ",DunnIndex
print "Sil score   : ",sil_sco
#plot_max_Sil_Score(Idata, n, Initial_Label, plot_label="Initial",sil_sco=sil_sco)
#plt.scatter(population[:,0],population[:,1], s=150, marker="x")
#plt.scatter(centers[:,0],centers[:,1], s=150, marker="o")
#plt.show()
H= 5 #int(input("Enter the size of neighbourhood mating pool: ")) #Size of mating pool
data=population
final_label=[]
print "############################################"

TDS = [0]
generation=0
self_pop = None
zdt_definitions = None
plotter = None
problem = None
evolution = None
while TDS:
    TDS = []
    print "-------------------------------------------------------------------------------"
    print "Generation No., data size , shape  : ",generation, data.size, population.shape
    if not data.size==0:
        for i in range(len(population)):
            MatingPool = np.arange(len(population))  # Generate mating pool
            MatingPool=np.delete(MatingPool,i)
            print "Mating Pool : ", MatingPool,type(MatingPool)  # Mating pool of each neuron
            nsol=Generate(MatingPool,  H, population,i,CR=0.8,F=0.8,mutation_prob=0.6,eta=20) #Generate new solution/Children
            print "Nsol generated using mating Pool : ", nsol
            cluster_center = np.reshape(nsol, (-1, Idata.shape[1]))
            print "new solutiojn in cluster center form : ", cluster_center
            cluster = KMeans(n_clusters=n, init=cluster_center)
            cluster.fit(Idata)
            new_center=cluster.cluster_centers_
            label = cluster.predict(Idata)
            new_ss = silhouette_score(Idata, label)
            print "new solution after clustering  : ", new_center
            new_pbm = cal_pbm_index(n, Idata, new_center, label)
            a = new_center.tolist()
            new_sol=[]
            for j in range(len(a)):
                for k in range(len(Idata[0])):
                    new_sol.append(a[j][k])
            print "new solution in list form : ", new_sol
            #new_di = calculate_dunn_index(Idata,label,n_cluster=n)
            print "NewSS",new_ss
            print "NewDI", new_pbm
            U_population,objectives,self_population,zdt_definitions,plotter,problem,evolution,Final_label = Select(population,sil_sco,DunnIndex,new_sol,new_pbm,new_ss,generation,self_pop,zdt_definitions,plotter,problem,evolution,Final_label,label)
            self_pop=self_population
            population= np.array(U_population)
        break

    num_rows, num_cols = data.shape
    b = data.tolist()
    for j in range(num_rows):
        a=population[j].tolist()
        if a not in b:
            TDS.insert(j,a)
    print "TDS",np.array(TDS)
    data = np.array([])
    data=np.array(TDS)
    generation+=1
    Iflag = generation
print "Returned pop",population
print "Objectives  (Sil. score, PBM)  : ",objectives
return_ss = [item[0] for item in objectives]
return_di = [item[1] for item in objectives]
print "PBM : ",return_di
print "SS  : ",return_ss
print "Max. PBM : ",max(return_di)
print "Max SS  : ",max(return_ss)
#x= return_ss.index(max(return_ss))
y= return_di.index(max(return_di))
#updated_label1=list(Initial_Label[x])
updated_label2=list(Initial_Label[y])
#print "ARI1",adjusted_rand_score(actual_label,updated_label1)

print "ARI2",adjusted_rand_score(actual_label,updated_label2)
all_ari=[]
for i in range (len(return_ss)):
    x=adjusted_rand_score(actual_label, Final_label[i])
    print "ARI ",i, x
    all_ari.insert(i,x)
print "Max ARI",max(all_ari)

