'''
Read in a specified number of n documents from enron text file
Create a set of words for each document
'''
import csv
import timeit
import math
import sys
import numpy as np
import fractions
import heapq
import random
'''
To-do:
Test when docs are the same
when all docs are the same
when there are mising words

(1) Can you identify where most of the time is being spent? 
You might be able to speed things up by using libraries like numpy.  
This could be particularly problematic if you are using these kinds of 
built in methods in one approach but not the other.  
(2) Do you still see this behavior for larger sets?
1. It's worth looking at how long different steps of your algorithm are taking.  
This can help you to narrow down what parts to look at.  
If you are using Python, loops can be particularly slow.
 
2. Can you match up your actual running times with your theoretical results?  
For example if you increase the number of minhashes n and look at how long it 
takes to create the signature matrix, does this match with you theoretical 
runtime estimate?
 
3. Running on the whole dataset may take awhile, but I would expect that 
you should be able to run on something like at least 10,000 documents 
without waiting too long. 
 
4. Lastly, this is a good opportunity to really starting thinking about 
the relationship between running time and your implementation choices, 
but that isn't the everything in this assignment.  
Get you code working correctly first and then see what runtime 
challenges you can tackle.  You can certainly talk about these kinds 
of challenges and what your strategy was in your write-up for the assignment.
'''

'''
Function to read a user-specified number of documents from the enron text file, 
and tore them as a set of wordIds.
Return: list documents - a list of sets, each sets represents the words present in a document
        list docId - a list of pointers to the documents
'''

def readDocuments():
    print "------------------------- Read Text Documents from File ------------------------"
    numDoc = raw_input('Enter the number (less then the total number of documents) of documents to use: ')
    with open("docword.enron.txt") as f:
        totalDoc = int(float(f.next()))
        print totalDoc

    while ((numDoc.isdigit() == False) | (int(float(numDoc)) <= 0) | (int(float(numDoc)) > totalDoc)):
        print ("Inappropriate number entered. Please enter appropriate integer.")
        numDoc = raw_input('Enter the number of documents to use: ')

    numDoc = int(float(numDoc))
    docCount = 0
    curDoc = 0
    documents = []
    docId = []
    for d in range (numDoc): #create a set for each document read in
        documents.append(set())
    
    with open("docword.enron.txt") as f:
        for i in range(3):
            f.next()
        for line in f:
            #print line
            curLine = line.split()
            if (int(curLine[0]) > curDoc):      #if doc number greater than current, means moved to the next doc
                curDoc = int(curLine[0])        #change the current doc id
                docId.append(curDoc)
                docCount += 1                   #number of docs processed
            if (docCount > numDoc):             #after done with numDocs docs, break
                docId.pop(-1)
                break
            documents[docCount-1].add(int(float(curLine[1])))    
    print "Finished reading. Number of items in Document List is: " + str(len(documents))                             #document n is stored in index n-1
    return [documents,docId]

'''
Function to find the actual jaccard similarity between two documents
Param - list documents - list of set of words for each document
        list docId - list of the document Id of the files corresponding to the index in documents
'''
def JaccardSim(documents, docId, *argv): #optionally, doc1 and doc2 can be given - for the brute force part
    
    if argv: #if nonempty
        doc1 = argv[0]
        doc2 = argv[1]
    else:
        doc1, doc2 = raw_input("Enter the IDs of two documents you wish to compare separated by a space:").split()
        doc1 = int(float(doc1))
        doc2 = int(float(doc2))
    
    while doc1 not in docId or doc2 not in docId: #if there's no corresponding doc 
        print "At least one of the your document IDs is invalid."
        print docId
        doc1, doc2 = raw_input("Enter the IDs of two documents you wish to compare separated by a space:").split()
        doc1 = int(float(doc1))
        doc2 = int(float(doc2))
    
    doc1_index = docId.index(doc1) 
    doc2_index = docId.index(doc2) #finds the position of the set in the list of sets
    union = len(documents[doc1_index].union(documents[doc2_index]))
    intersection = len(documents[doc1_index].intersection(documents[doc2_index]))
    jac_sim = float(intersection)/union
    return jac_sim

'''
Function to find the k most similar documents using brute force and actual similarity
Param - list documents
        list docId
'''
def bruteForce(documents,docId,*argv):
    
    if argv: #if nonempty
        doc = argv[0]       #doc is the docId
        k = argv[1]         

    else:
        doc = raw_input("Enter the document you want to compute the k neighbors for: ")
        doc = int(float(doc))
        while doc not in docId : #if there's no corresponding doc 
            print "Your document IDs is invalid."
            doc = raw_input("Enter the document you want to compute the k neighbors for: ")
            doc = int(float(doc))
        k = raw_input("How many neighbors would you like to find? ")

        while ((k.isdigit() == False) | (int(float(k)) >= (len(docId))) | (int(float(k)) <= 0)):
            print ("Inappropriate k value chosen. Please enter appropriate integer value.")
            k = raw_input("How many neighbors would you like to find?")
        k = int(float(k))
    
    neighbors = []
    knn = []
    heapq.heapify(neighbors)
    similarity_count = 0
    average_sim = 0

    iterate = 0
    while (len(neighbors) < k):
        if (docId[iterate] != doc):
            testId = docId[iterate]
            testDoc = documents[iterate]
            jac_sim = JaccardSim(documents, docId, testId, doc)
            item = (jac_sim, testId)
            heapq.heappush(neighbors, item)
        iterate += 1
    while (iterate < len(docId)):
        if (docId[iterate] != doc):
            testId = docId[iterate]
            testDoc = documents[iterate]
            jac_sim = JaccardSim(documents, docId, testId, doc)
            item = (jac_sim, testId)
            heapq.heappushpop(neighbors, item)
        iterate += 1
    for each in neighbors:
        similarity_count += each[0]
        knn.append(each[1])
    
    average_sim = float(similarity_count)/len(knn)
    print "The average similarity of document " + str(doc) + " is " + str(average_sim)

    return average_sim

'''
Function to find the average similarity of all documents and their k neighbors
Param - documents
        docId
'''
def avg_avg_knn(documents,docId):
    k = raw_input("How many neighbors would you like to find? ")

    while ((k.isdigit() == False) | (int(float(k)) >= (len(docId))) | (int(float(k)) <= 0)):
        print("Inappropriate k value chosen. Please enter appropriate integer value.")
        k = raw_input("How many neighbors would you like to find?")
    k = int(float(k))
    start_time = timeit.default_timer()
    totalAvg = 0
    for i in range(len(docId)):
        cur = bruteForce(documents,docId,docId[i],k)
        totalAvg = totalAvg + cur

    avg_avg = float(totalAvg)/len(docId)
    print "###############################################################"
    print "Compute avg_avg knn time (brute force): " + str(timeit.default_timer() - start_time)
    print "By BRUTE FORCE, the average knn using k = " + str(k) + " similarity across ALL documents is " + str(avg_avg)
    print "###############################################################"

'''
Function to compute hash function
Param: x - original 'row' number in char
       a - random number from 0 to 1-x and is prime
       b - random number from 0 to 1-x
       w - number of unique words in the selected documents
'''

def minHash(x,a,b,w):
    newIndex = (a * x + b) % w
    return int(float(newIndex))

'''
Function to generate a signature matrix for all documents
'''
def getSigMatrix(documents):
    np.set_printoptions(suppress=True)
    print "----------------------- Generating Signature Matrix ------------------------"
    numWords = 0
    allWords = set.union(*documents)
    allWords = list(allWords)   #index of the wordId in list allWords is the original row number
    numWords = len(allWords)                      #total number of vocabulary
    print "Number of unique words in all files read: " + str(numWords)

    numRows = raw_input('Enter the number of rows for the signature matrix: ')
    while ((numRows.isdigit() == False) | (int(float(numRows)) <= 0) | (int(float(numRows)) > numWords)):
        print "Inappropriate row number. Please re-enter."
        numRows = raw_input('Enter the number of rows for the signature matrix: ')
    numRows = int(float(numRows))
    '''
    Generate appropriate number of hash functions
    '''
    start_time = timeit.default_timer()
    #Compute hash functions such that h(x) = (ax+b mod w), and a is a prime number less then w.
    w = int(numWords)
    #generate a list of relatively prime for values of a
    a = []
    np.random.seed(4)

    for num in range(numRows):
        add = False
        while (add == False):
            aVal = np.random.randint(1,w)
            if (fractions.gcd(aVal,w)==1):
                add = True
                a.append(aVal)
    np.random.seed(411)
    b = list(np.random.choice(range(w),numRows, replace = False))

    '''
    Generate signature matrix
    '''
    #Initialize a signature matrix with infinity in all slots
    sigMatrix = np.empty((numRows, len(documents),))
    sigMatrix[:] = float("inf")
    print sigMatrix
    wordcount = 0
    hashVal = [0] * numRows
    for x in range (len(allWords)):          #for every row in character matrix
        wordcount+=1
        
        for j in range(numRows):   #compute hash values for row 'x' for each hash fcn and store in list hashVal
            hashVal[j] = minHash(x,a[j%len(a)],b[j],w)


        for doc in range(len(documents)):      #for every document, check if the word is in its set of words.
            if (allWords[x] in documents[doc]):
                for y in range(numRows):
                    if (hashVal[y] < sigMatrix[y][doc]):     #y^th row, doc^th column
                        sigMatrix[y][doc] = hashVal[y]
    print "##########################Signature Matrix#############################"
    print sigMatrix
    print "Compute getSigmatrix time: " + str(timeit.default_timer() - start_time)
    return sigMatrix



############################################################################################################
############################################################################################################
############################################# L  S  H ######################################################
############################################################################################################
############################################################################################################

'''
Function to ESTIMATE jaccard similarity from the signature matrix
'''
def estJaccard(sigMatrix, docId, *argv):
    if argv: #if nonempty
        doc1 = argv[0]       #doc is the docId
        doc2 = argv[1]         

    else:
        doc1 = 0
        doc2 = 0
        doc1_index = 0
        doc2_index = 0

        doc1, doc2 = raw_input("Enter the IDs of two documents you wish to compare separated by a space:").split()
        doc1 = int(float(doc1))
        doc2 = int(float(doc2))
        while doc1 not in docId or doc2 not in docId: #if there's no corresponding doc 
            print "At least one of the your document IDs is invalid."
            doc1, doc2 = raw_input("Enter the IDs of two documents you wish to compare separated by a space:").split()
            doc1 = int(float(doc1))
            doc2 = int(float(doc2))
    
    doc1_index = docId.index(doc1) 
    doc2_index = docId.index(doc2) #finds the position of the set in the list of sets
    
    ############################## Actual Calculation ###########################
    union = len(sigMatrix)
    intersection = 0
    for row in sigMatrix:
        if row[doc1_index] == row[doc2_index]:
            intersection += 1

    jac_est = float(intersection)/union
    return jac_est

'''
Function that preprocess the signature matrix and generates a list of dictionaries of potentially similar documents
'''
def LSH_buckets(sigMatrix, docId, numRows):
    print "---------------Preprocessing the data by banding for LSH --------------------"
    r = raw_input("For BANDING, enter the number of rows (r) in each band:")
    while ((r.isdigit() == False) | (int(float(r)) <= 0) | (int(float(r)) > numRows)| (numRows % (int(float(r))) != 0 )):
        print "Inappropriate number of rows. Please re-enter."
        print "Make sure r divides " + str(numRows)
        r = raw_input("Enter the number of rows in each band:")
    r = int(float(r))
    b = int(float(numRows/r))
    list_of_dicts = []
    flipped_sigMatrix = sigMatrix.T
    start = 0
    end = r
    start_time = timeit.default_timer()
    for times in range(b): #b times
        #print "Now processing band " + str(times)
        hashes = {} # tuples are unique keys, docIds are values
        for i in range(len(flipped_sigMatrix)): #look at each column
            #band_counter = 0 #counts # of elements in tuple
            #bucket_key = []
            bucket = []
            bucket_key = tuple(flipped_sigMatrix[i][start:end])
            hashes.setdefault(bucket_key,[]).append(docId[i])
        list_of_dicts.append(hashes)
        start += r
        end += r
    #print list_of_dicts
    print("Compute dictionary time: " + str(timeit.default_timer() - start_time))
    return list_of_dicts
        
'''
Function that calculates the avg kNN similarity for a given document
'''
def LSH(list_of_dicts, docId, sigMatrix, *argv):
    if argv: #if nonempty
        doc_to_examine = argv[0]       #doc is the docId
        k = argv[1]         
    else:
        doc_to_examine = raw_input("Enter the ID of the document you wish to examine:")
        while ((doc_to_examine.isdigit == False) | (int(float(doc_to_examine)) <= 0) | (int(float(doc_to_examine)) not in docId)):
            print "Inappropriate document ID. Please re-enter."
            doc_to_examine = raw_input("Enter the ID of the document you wish to examine:")
        doc_to_examine = int(float(doc_to_examine))
    
        k = raw_input("Enter the number of nearest neighbors:")
        while ((k == False) | (int(float(k)) <= 0) | (int(float(k)) > len(docId))):
            print "Inappropriate document ID. Please re-enter."
            k = raw_input("Enter the ID of the document you wish to examine:")
        k = int(float(k))
    
    candidates = set() #a set of potential candidates
    bigList = []
    for dict in list_of_dicts:
        bigList.append(dict.values())
    #bigList = set.union(*bigList) #creates list of buckets
    for list_of_buckets in bigList:
        for buck in list_of_buckets:
            if doc_to_examine in buck and len(buck) > 1:
                for each in buck:
                    if each == doc_to_examine:
                        continue
                    else:
                        candidates.add(each) #creates a set of candidates
    
    randCand = 0
    if len(candidates) < k:
        randCand = 1
    #go through and keep a priority queue of the k biggest estimated JacSims
    while len(candidates) < k:
        np.random.seed(1229)
        toAdd = random.choice(docId)
        if toAdd != doc_to_examine:    
            candidates.add(toAdd) 

    candidates = list(candidates)
    k_neighbors = []
    heapq.heapify(k_neighbors)
    est_sim_count = 0
    average_est_sim = 0
    for cand in candidates[0:k]:
        est_jac = estJaccard(sigMatrix, docId, doc_to_examine, cand)
        node = (est_jac, cand)
        heapq.heappush(k_neighbors, node) #creates a priority queue with k elements. 
    for cand2 in range(k,len(candidates)):
        est_jac = estJaccard(sigMatrix, docId, doc_to_examine, candidates[cand2])
        node2 = (est_jac, candidates[cand2])
        heapq.heappushpop(k_neighbors, node2)
    #print k_neighbors
        
    for boom in k_neighbors:
        est_sim_count+=boom[0]
    average_est_sim = float(est_sim_count)/k
    #print "####################################################################"
    print "Average estimate Jaccard similarity for doc " +str(doc_to_examine) + " is " + str(average_est_sim)
    #print "####################################################################"
    return [average_est_sim, randCand]



def avg_avg_lsh(list_of_dicts, docId, sigMatrix):
    print "------------Running LSH to find the average similarity of all documents and their k neighbors -------------"
    k = raw_input("How many neighbors would you like to find? ")

    while ((k.isdigit() == False) | (int(float(k)) >= (len(docId))) | (int(float(k)) <= 0)):
        print("Inappropriate k value chosen. Please enter appropriate integer value.")
        k = raw_input("How many neighbors would you like to find?")
    k = int(float(k))
    start_time = timeit.default_timer()
    totalAvg = 0
    doc_with_randCand = 0
    for i in range(len(docId)):
        cur = LSH(list_of_dicts, docId, sigMatrix, docId[i], k)
        totalAvg = totalAvg + cur[0]
        doc_with_randCand += cur[1]
        
    avg_avg = float(totalAvg)/len(docId)
    print "################################################################"
    print "Compute avg_avg lsh time (LSH): " + str(timeit.default_timer() - start_time)
    print "The number of documents with randomly chosen neighbors is " + str(doc_with_randCand)
    print "By LSH, the average similarity across ALL documents is " + str(avg_avg)
    print "################################################################"


def main():
    #Task 1. read file
    start_time = timeit.default_timer()
    data = readDocuments()           #data[0] is documents, data[1] is docId
    print("Read time: " + str(timeit.default_timer() - start_time))
    
    
    #Task 1.5 Enter two documents
    doc1, doc2 = raw_input("Enter the IDs of two documents you wish to compare separated by a space:").split()
    doc1 = int(float(doc1))
    doc2 = int(float(doc2))
    
    while doc1 not in data[1] or doc2 not in data[1]: #if there's no corresponding doc 
        print "At least one of the your document IDs is invalid."
        doc1, doc2 = raw_input("Enter the IDs of two documents you wish to compare separated by a space:").split()
        doc1 = int(float(doc1))
        doc2 = int(float(doc2))
    
    doc1_index = data[1].index(doc1) 
    doc2_index = data[1].index(doc2) #finds the position of the set in the list of sets
    
    
    #Task 2. Actual Jaccard
    print "---------------Computing Actual Jaccard Similarity between two files------------"
    jac_sim = JaccardSim(data[0], data[1], doc1, doc2)
    print "##########################################################################################"
    print "Actual Jaccard similarity between the given documents is: " + str(jac_sim)
    print "##########################################################################################"
    
    #Task 3. Signature matrix
    start_time = timeit.default_timer()
    sig = getSigMatrix(data[0])
    print "Get SignatureMatrix time: " + str(timeit.default_timer() - start_time)
    
    #Task 4. Estimate the Jaccard sim for the two docs using sig matrix
    print "----------------------- Estimating Jaccard Similarity ------------------------"
    jac_est = estJaccard(sig, data[1], doc1, doc2)
    print "##########################################################################################"
    print "Estimated Jaccard similarity of the two docs is " + str(jac_est)
    print "##########################################################################################"
    
    
    #Task 5. Enter r, preprocess by banding
    list_of_dicts = LSH_buckets(sig, data[1], len(sig))
    
    #Task 6. Enter k, find the k nearest neighbors
    print "---------------Using brute force to find the similarity of all files with their knn --------"
    start_time = timeit.default_timer()
    avg_avg_knn(data[0],data[1])
    #print("Compute avg_avg knn time (brute force): " + str(timeit.default_timer() - start_time))

    print "---------------Using LSH to find the similarity of all files with their knn --------"
    start_time = timeit.default_timer()
    avg_avg_lsh(list_of_dicts, data[1], sig)
    #print("Compute avg_avg lsh time (LSH): " + str(timeit.default_timer() - start_time))

    done = False

    while (done == False):
        print "-------Commands--------"
        print "To compute Actual Jaccard Similarity: Enter 1."
        print "To compute Estimated Jaccard Similarity: Enter 2."
        print "To compute brute force Similarity: Enter 3."
        print "To recompute signature matrix: Enter 4."
        print "To compute the avg of avg similarity over all docs (brute force): Enter 5."
        print "To compute the avg of avg similarity over all docs (LSH): Enter 6."


        print "To quit, enter q."
        command = raw_input('Enter command: ')
        if (command == "1"):
            print "---------------Computing Actual Jaccard Similarity between two files------------"
            #start_time = timeit.default_timer()
            JaccardSim(data[0],data[1])
            print "##########################################################################################"
            print "Actual Jaccard similarity between the given documents is " + str(jac_sim)
            print "##########################################################################################"
            #print("Compute actual Jaccard sim time: " + str(timeit.default_timer() - start_time))

        elif (command == "2"):
            print "----------------------- Estimating Jaccard Similarity ------------------------"
            #start_time = timeit.default_timer()
            jac_est = estJaccard(sig, data[1])
            print "##########################################################################################"
            print "Estimated Jaccard similarity of the two docs is " + str(jac_est)
            print "##########################################################################################"
            #print("Compute estimate Jaccard sim time: " + str(timeit.default_timer() - start_time))

        elif (command == "3"):
            print "--------------------- Using brute force to find the k nearest neighbors---------------------"
            #start_time = timeit.default_timer()
            kneighbors = bruteForce(data[0],data[1])
            print "##########################################################################################"
            print "The Average Jaccard Similarity with k neighbors is: " + str(kneighbors)
            print "##########################################################################################"
            #print("Compute bruteforce time: " + str(timeit.default_timer() - start_time))

        elif (command == "4"):
            sig = getSigMatrix(data[0])

        elif (command == "5"):
            print "---------------Using brute force to find the similarity of all files with their knn --------"
            avg_avg_knn(data[0],data[1])
            
        elif (command == "6"):
            print "---------------Using LSH to find the similarity of all files with their knn --------"
            avg_avg_lsh(list_of_dicts, data[1], sig)

        else:
            done = True





main()