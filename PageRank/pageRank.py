'''
DataMining Homework 4
Page Ranks and Protein Interactions
Author: Vianne Gao  
'''

import numpy as np
import sys

'''
Function to preprocess the data and remove any dead ends.
Return - preprocesses data with no deadends
'''
def preprocess():

    #Convert txt file into a n*2 matrix
    data = np.genfromtxt ('wnt_edges.txt', skip_header = 1, dtype="|S6")
    #convert to list for easy removal of dead ends
    dataList = data.tolist()
    tail = data[:,0]
    head = data[:,1]

    #remove all the dead ends, which essentially are vertices in head but not in tail.
    while True:
        n = len(dataList)
        dataList = [x for x in dataList if x[1] in tail]
        tail = np.asarray(dataList)[:,0]
        if n == len(dataList):
            break

    data = np.asarray(dataList)

    print 'The number of unique proteins after removing dead ends is: ' + str(np.unique(data).size)

    return data

'''
Function that converts data into a transition matrix.
Return - the transition matrix and the list of unique nodes
'''
def toTransitionMatrix(data):

    #get the number of unique nodes from data
    uniqueData = np.unique(data)

    #create a zero matrix for the transition matrix
    matrix = np.zeros((uniqueData.size , uniqueData.size))

    for i in range(uniqueData.size):

        links = [x for x in data if x[0] == uniqueData[i]]
        links = np.asarray(links)
        
        for j in range(uniqueData.size):
            if uniqueData[j] in links[:,1]:
                matrix[j,i] = 1.0/links[:,1].size

    return [matrix,uniqueData]

'''
Function to compute pageRand using the power method
Return - stationary vector containing the page ranks for each page
'''
def computeRank(matrix, beta, iters):

    #initial vector
    v = np.zeros((matrix.shape[0],1)) + 1.0/matrix.shape[0]
    #unit vector
    e = np.zeros((matrix.shape[0],1)) + 1.0

    #iterate using the power method
    for i in range(iters):
        v = beta * np.dot(matrix, v) + (1-beta) * e/matrix.shape[0]

    return v

def main(beta,iters):
    #Check whether beta and iters are reasonable
    if float(beta) > 1 or float(beta) < 0:
        print "You entered invalid beta value. Using beta = 1."
        beta = 1
    if float(iters) < 0:
        print "Invalid iteration, using iters = 10."
        iters = 10
    beta = float(beta)
    iters = int(float(iters))
    preprocessData = preprocess()
    matrix = toTransitionMatrix(preprocessData)
    v = computeRank(matrix[0], beta, iters)
    
    #Find the 5 proteins with the highest pageRanks
    v = v.flatten()
    ind = np.argsort(v)[::-1]
    v = np.sort(v, axis=None)[::-1]
    print "Proteins with the highest pageRank:"
    for i in range (5):
        print "Protein: " + str(matrix[1][ind[i]]) + "  pageRank: " + str(v[i])
    
main(sys.argv[1], sys.argv[2])