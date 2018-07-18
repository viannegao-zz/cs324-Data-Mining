'''
Data Mining Assignment 6
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
Function to initialize centroids randomly by selecting k data points
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
Sub-samples of size 500 (5% of the data) are sampled for 30 times from the original 
data points. Each sub-sample is clustered using the k-means algorithm with the initial 
centroids chosen randomly. The centroids are clustered to generate the initial centroids.
(Documented in README.md)
Param: int k: number of clusters
       array set_train: numpy matrix with each datapoint as a row
'''
def refined_init(k,set_train):
    cent_data = []
    print '--------- Refining initial centroids ----------'
    for i in range (30): #Sample 5% of the data and cluster, for 30 times
        sub_train = np.asarray(set_train[np.random.randint(set_train.shape[0], size=500), :])
        #Cluster the subset of data and add centroids to an array
        init_centroids = rand_init(k,sub_train)
        classify = K_Means(k,sub_train)
        classify.cluster(init_centroids,0)
        cent_data.append(classify.centroids.values())
    
    #Cluster the centroids obtained previously
    cent_data = list(itertools.chain.from_iterable(cent_data))
    cent_data = np.asarray(cent_data)
    init_centroids = rand_init(k,cent_data)
    classify_cent = K_Means(k,cent_data)
    classify_cent.cluster(init_centroids,0)
    print '---------- Clustering Data -----------'
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
        while not done:         # while there are still empty clusters, repeat
            done = True
            for key, val in self.classifId.items():
                if val == []:   #if cluster empty, replace centroid
                    #print 'Empty cluster: ' + str(key)
                    done = False
                    #replace with datapoint furthest away from its centroid
                    replaceCentId = alldist.index(max(alldist))
                    alldist.pop(replaceCentId)
                    replaceCent = self.data[replaceCentId]
                    self.centroids[key] = replaceCent
            if done:
                break
            # Reassign the datapoints to its closest centroid
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
    def cluster(self,init_centroids,print_sse=1):
        self.centroids = {}     #keep track of the centroid of each cluster labeled 1 to k
        for i in range(self.k): #initially, use the initialized centroid
            self.centroids[i] = init_centroids[i]

        converged = False
        while not converged: #repeat until the centroids do not change
            
        ### 1.Assign points to closest centroid ###
            self.classifications = {} #keep track of datapoints assigned to each cluster
            self.classifId = {}       #keep track of index of data assigned to each cluster
            sse_List = []
            for j in range(self.k):
                self.classifications[j] = [] #keep track which data are assigned to each centroid
                self.classifId[j] = []       #keep track of the id of data assigned to each centroid
                self.distList = []

            for entry in range (len(self.data)): #assign datapoint to closest centroid and record distance
                #compute distance to each centroid stored in dict centroids
                distances = [np.linalg.norm(self.data[entry]-self.centroids[cent]) for cent in self.centroids]
                newCent = distances.index(min(distances)) #assign entry to the closest centroid
                self.distList.append(min(distances))
                self.classifications[newCent].append(self.data[entry])
                self.classifId[newCent].append(entry)
            self.handleEmptyCluster()

        ######## 2.Recomputing centroids ########
            old_cents = dict(self.centroids)     #keep track of the previous centroids 
            for newCent in self.classifications: #recompute each centroid by averaging the data in cluster
                self.centroids[newCent] = np.average(self.classifications[newCent],axis=0)
                ### Compute SE for the cluster ###
                for datapoint in self.classifId[newCent]: #for cur centroid, compute SE for its datapoints
                    se = np.sum((self.data[datapoint] - self.centroids[newCent])**2)
                    sse_List.append(se)
            sse = sum(sse_List)
            if print_sse == 1:
                print 'Current SSE: ' + str(sse)
            
            converged = True               #first assume convergence
            for cent in self.centroids:    #compute the change in each centroid
                oldCent = old_cents[cent]
                curCent = self.centroids[cent]
                if np.sum(curCent-oldCent) > 0.0:
                    converged = False      #if at least one centroids changed, repeat iteration

def main(k, init):
    k = int(k)
    set_train = np.loadtxt('number_data.txt',delimiter=',') #read data into matrix
    set_label = np.loadtxt('number_labels.txt')
    
    # Initialize centroids
    if init == 'rand':
        print '-------- using rand initialization ---------'
        init_centroids = rand_init(k,set_train)
    elif init == 'cheat':
        print '----- using cheat initialization -----'
        init_centroids = cheat_init(set_train,set_label)
        k = 10
    else:
        print '----- using other (refined) initialization -----'
        init_centroids = refined_init(k,set_train)
    
    # Cluster the datapoints
    classify = K_Means(k,set_train)   #create a K_Means object
    classify.cluster(init_centroids)

main(sys.argv[1],sys.argv[2])





