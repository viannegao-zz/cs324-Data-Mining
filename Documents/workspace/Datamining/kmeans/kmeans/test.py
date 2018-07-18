'''
Data Mining Assignment 6 - Analysis Questions
K-means Clustering Algorithm for MNIST Digits
Program that clusters datapoints into k clusters and print to the screen 
the value of this objective function at the end of each iteration of the algorithm
Author: Vianne Gao
'''
import numpy as np
import scipy as sp
import sys
import timeit
import itertools
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import random

'''
Function to initialize centroids randomly
Param: int k: number of clusters
       array set_train: numpy matrix with each datapoint as a row
'''
def rand_init(k,set_train):
    init_centroids = []
    cent_index = random.sample(xrange(0,len(set_train)),k)
    for point in cent_index:
        init_centroids.append(set_train[point])
    return np.asarray(init_centroids)

'''
Function to generate centroids using the refined initialization algorithm by Fayyad et.al
Param: int k: number of clusters
       array set_train: numpy matrix with each datapoint as a row
'''
def refined_init(k,set_train):
    #generate a list of 10 list of k centroid arrays
    cent_data = []
    for i in range (30):
        sub_train = np.asarray(set_train[np.random.randint(set_train.shape[0], size=500), :])
        #Cluster the subset of data and add centroids to an array
        init_centroids = rand_init(k,sub_train)
        classify = K_Means(k,sub_train)
        classify.cluster(init_centroids,0)
        cent_data.append(classify.centroids.values())
    
    #Cluster the centroid datapoints
    cent_data = list(itertools.chain.from_iterable(cent_data))
    cent_data = np.asarray(cent_data)
    init_centroids = rand_init(k,cent_data)
    classify_cent = K_Means(k,cent_data)
    classify_cent.cluster(init_centroids,0)
    return np.asarray(classify_cent.centroids.values())

'''
Function to cheat the initialization
'''
def cheat_init(set_train,set_label):
    centroids = {}
    groups = {}
    for i in range (1,11):
        groups[i] = set_train[(i-1)*1000:i*1000]
    for g in groups: #recompute each centroid by taking the avg of all data in the cluster
        centroids[g] = np.average(groups[g],axis=0)
    return np.asarray(centroids.values())

'''
K-means Algorithm
Clusters provided datapoints into k clusters using euclidean distance.
Objective is to minimize SSE. Prints SSE after each iteration.
Param: int k: number of clusters desired
       array data: numpy matrix with each datapoint as a row
'''
class K_Means:
    def __init__(self, k, data):
        self.k = k
        self.data = data

    #Function to handle empty clusters by replacing the empty centroid 
    #   with the most distant datapoints
    def handleEmptyCluster(self):
        alldist = self.distList[:]
        done = False
        while not done:     #while there are still empty clusters, repeat
            done = True
            for key, val in self.classifId.items():
                if val == []:   #if cluster empty, replace centroid
                    print 'Empty cluster: ' + str(key)
                    done = False
                    #replace with datapoint furthest away from its centroid
                    replaceCentId = alldist.index(max(alldist))
                    alldist.pop(replaceCentId)
                    replaceCent = self.data[replaceCentId]
                    self.centroids[key] = replaceCent
            if done:
                break
            #Reclassify the datapoints to its closest centroid
            self.classifications = {}
            self.classifId = {}
            for i in range(self.k):
                self.classifications[i] = [] #keep track which data are assigned to each centroid
                self.classifId[i] = []       #keep track of the id of data assigned to each centroid
                self.distList = []           
            for entry in range(len(self.data)): #assign each datapoint to closest centroid and record distance
                distances = [np.linalg.norm(self.data[entry]-self.centroids[cent]) for cent in self.centroids]
                newCent = distances.index(min(distances)) #assign entry to the closest centroid
                self.distList.append(min(distances))
                self.classifications[newCent].append(self.data[entry])
                self.classifId[newCent].append(entry)

    #Function to cluter all datapoints into k clusters using kmeans
    def cluster(self,init_centroids,printsse=1):
        self.sse_iter=[]
        self.centroids = {}     #keep track of the centroid of each cluster labeled 1 to k
        for i in range(self.k): #initially, use the initialized centroid
            self.centroids[i] = init_centroids[i]

        converged = False
        while not converged: #repeat until the centroids do not change
            
        ### 1.Assign points to closest centroid ###
            self.classifications = {} #keep track of datapoints assigned to each cluster
            self.classifId = {}       #keep track of index of data assigned to each cluster
            sse_List = []
            cse_list = []
            for j in range(self.k):
                self.classifications[j] = [] #keep track which data are assigned to each centroid
                self.classifId[j] = []       #keep track of the id of data assigned to each centroid
                self.distList = []

            for entry in range (len(self.data)): #assign each datapoint to closest centroid and record distance
                #compute distance to each centroid stored in dict centroids
                distances = [np.linalg.norm(self.data[entry]-self.centroids[cent]) for cent in self.centroids]
                newCent = distances.index(min(distances)) #assign entry to the closest centroid
                self.distList.append(min(distances))
                self.classifications[newCent].append(self.data[entry])
                self.classifId[newCent].append(entry)
            self.handleEmptyCluster()

        ######## 2.Recomputing centroids ########
            old_cents = dict(self.centroids)     #keep track of the previous centroids 
            for newCent in self.classifications: #recompute each centroid by taking the avg of all data
                clus_error = []
                self.centroids[newCent] = np.average(self.classifications[newCent],axis=0)
                for datapoint in self.classifId[newCent]:
                    se = np.sum((self.data[datapoint] - self.centroids[newCent])**2)
                    sse_List.append(se)
                    clus_error.append(se)
                cse = sum(clus_error)
                cse_list.append(cse)
            sse = sum(sse_List)
            self.sse_iter.append(sse)
            if printsse == 1:
                print 'Current SSE: ' + str(sse)
            converged = True                        #First assume convergence

            for cent in self.centroids:             #compute the change in each centroid
                oldCent = old_cents[cent]
                curCent = self.centroids[cent]
                if np.sum(curCent-oldCent) > 0.0:
                    converged = False               #If at least one centroids changed repeat iteration
        if printsse == 1:
        #print the sse within each cluster
            print 'Cluster SSE: ' + str(cse_list)

def display_image(data):
    data = np.array(data)
    data = np.reshape(data,(-1,28))
    plt.imshow(data)
    plt.show()

def main(k, init,q=None):
    q = int(q)
    k = int(k)
    set_train = np.loadtxt('number_data.txt',delimiter=',') #read data into matrix
    set_label = np.loadtxt('number_labels.txt')
    print len(set_train)
    
    if init == 'rand':
        print '-------- using rand initialization ---------'
        init_centroids = rand_init(k,set_train)
    elif init == 'cheat':
        print '----- using cheat initialization -----'
        init_centroids = cheat_init(set_train,set_label)
    else:
        print '----- using other (refined) initialization -----'
        start_time = timeit.default_timer()
        init_centroids = refined_init(k,set_train)
        print "Initializing refined centroid: " + str(timeit.default_timer() - start_time)
    
    start_time = timeit.default_timer()
    classify = K_Means(k,set_train)
    classify.cluster(init_centroids)
    print "Running kmeans: " + str(timeit.default_timer() - start_time)
    
    print '------------Result------------'

    if q == 1:
        sse_iter = classify.sse_iter
        plt.plot(np.arange(len(sse_iter)),sse_iter,lw=1.5,color='red')
        plt.title('SSE After Every Iteration (k=10, rand initialization)')
        plt.ylabel('SSE')
        plt.xlabel('Iteration')
        plt.show()
    if q == 2: #k= 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
        final_sse_rand = [28292143772.6,25538217210.6,22772415840.7,21350039583.1,20425557608.9,
            19920181027.4,19316761545.3,18867348066.0,18554324221.2,18152317980.2,17920375475.6]
        final_sse_other = [28319485741.0,25580118008.8,22794602733.8,21381839304.4,20482758548.6,
            19781822667.7,19301182684.2,18843019090,18601244163.4,18252226605.8,18082918131.9]
        ylist = [5,10,20,30,40,50,60,70,80,90,100]
        # plt.plot(ylist,final_sse_rand,lw=1.5,color='red',label='random')
        # plt.title('SSE using a range of K values (random initialization)')
        # plt.ylabel('SSE')
        # plt.xlabel('K')
        # plt.show()
        plt.plot(ylist,final_sse_other,lw=1.5,color='blue',label='other')
        plt.title('SSE using a range of K values (refined initialization)')
        plt.ylabel('SSE')
        plt.xlabel('K')
        plt.show()        
        # k5=[28292143814.2]
        # k10=[25580118008.8]
        # k20=22794602733.8
        # k30=21381839304.4
        # k40=20482758548.6
        # k50=19781822667.7
        # k60=19301182684.2
        # k70=18843019090.0
        # k80=18601244163.4
        # k90=18252226605.8
        # k100=[18082918131.9]


        # k5 = [28292143772.6, 28292194082.5]
        # k10 = [25687238195.2,25589756827.0,25538217210.6]
        # k20 = [22772415840.7,22783162740.3]
        # k30=[21350039583.1,21359351047.6]
        # k40=[20425557608.9]
        # k50=[19920181027.4]
        # k60=[19316761545.8]
        # k70=[18867348066.0]
        # k80=[18554324221.2]
        # k90=[18152317980.2]
        # k100=[17920375475.6]

    if q == 4:
        for key, value in classify.classifId.items():
            counter = Counter()
            print 'cluster ' + str(key)
            cur_count = [set_label[id] for id in value]
            counter.update(cur_count)
            print counter
    # Display the images
    if q == 4:
        for i in range(k):
            display_image(classify.centroids.values()[i])

main(sys.argv[1],sys.argv[2],sys.argv[3])





