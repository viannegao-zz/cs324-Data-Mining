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
    
    # print "Document List: " + str(documents)                             #document n is stored in index n-1
    # print "Document Id: " + str(docId)                                 #Actual docId
    print "Number of items in Document List: " + str(len(documents))                             #document n is stored in index n-1
    print "Number of items in Document Id: " + str(len(docId))                                 #Actual docId

    return [documents,docId]
    '''
    Created set of wordid in each document.
    Created a list of docid to keep track
    '''


'''
Function that computes the exact Jaccard similarity between two sets.
Param: list - documents
       list - docId
Return: float - Exact Jaccard similarity between two documents
'''
def actualJaccard(documents,docId,*argv):
    print "---------------------Actual Jaccard Similarity----------------------------"
    '''
    other
    '''
    if argv: #if nonempty
        doc1 = argv[0]
        doc2 = argv[1]



    else:
        ##########
        #Id docs not ordered or all present, then need to find documents via list docId
        ##########
        doc1 = raw_input('Enter the Id of the first document: ')        #doc1 is sotred in index doc1 - 1 in documents
        while ((doc1.isdigit() == False) | ((int(float(doc1)) in docId) == False)):
            print "Inappropriate docId. Please re-enter."
            doc1 = raw_input('Enter the Id of the first document: ')

        doc2 = raw_input('Enter the Id of the second documents: ')
        while ((doc2.isdigit() == False) | ((int(float(doc2)) in docId) == False)):
            print "Inappropriate docId. Please re-enter."
            doc2 = raw_input('Enter the Id of the second document: ')
    
    doc1 = int(float(doc1))
    doc2 = int(float(doc2))

    # doc1, doc2 = raw_input("Enter the IDs of two documents you wish to compare separated by a space:").split()
    # doc1 = int(float(doc1))
    # doc2 = int(float(doc2))
    # while doc1 not in docId or doc2 not in docId: #if there's no corresponding doc 
    #     print "At least one of the your document IDs is invalid."
    #     doc1, doc2 = raw_input().split("Enter the IDs of two documents you wish to compare separated by a space:")
    #     doc1 = int(float(doc1))
    #     doc2 = int(float(doc2))
    
    print "doc1_index =  " + str(docId.index(doc1)) 
    print "doc2_index =  " + str(docId.index(doc2))
    
    doc1_index = docId.index(doc1) 
    doc2_index = docId.index(doc2) #finds the position of the set in the list of sets
    intersect = len((documents[doc1_index]).intersection(documents[doc2_index]))
    union = len((documents[doc1_index]).union(documents[doc2_index]))
    jac_sim = float(intersect)/union
    print "Intersection = " + str(intersect)
    print "Union = " + str(union)
    print "Actual Jaccard Similarity between " + str(doc1) + " and " + str(doc2) + " is: " + str(jac_sim)
    return jac_sim
    '''
    Returns the actual Jaccard Similarity
    '''

'''
Function to find the k most similar documents using brute force
Param - documents
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
    

    '''
    other way
    '''
    # neighbors = []
    # heapq.heapify(neighbors)
    # similarity_count = 0
    # average_sim = 0

    # for item in range(k):
    #     if item == docId.index(doc):
    #         continue #no need to compare to itself
    #     else:
    #         jac_sim=actualJaccard(documents, docId, doc, docId[item])
    #         heapq.heappush(neighbors, jac_sim) #creates a priority queue with k elements. 
    # for item2 in range(k,len(docId)):
    #     if item2 == docId.index(doc):
    #         continue
    #     else:
    #         jac_sim=actualJaccard(documents, docId, doc, docId[item2])
    #         if jac_sim > neighbors[0]:
    #             heapq.heapreplace(neighbors, jac_sim)
    # for each in neighbors:
    #     similarity_count+=each
    
    # average_sim = float(similarity_count)/len(neighbors)
    # print "The Average Jaccard Similarity with k neighbors is: " + str(average_sim)
    # return average_sim
    
    '''
    Find k neighbors
    '''
    neighbors = []
    check_index = docId.index(doc)
    check_doc = documents[check_index]
    
    i = 0
    while (len(neighbors) < k):
        if (docId[i] != doc):        #if not the file itself
            testDocId = docId[i]
            testDoc = documents[i]
            
            intersect = len((testDoc).intersection(check_doc))
            union = len((testDoc).union(check_doc))
            jac_sim = float(intersect)/union
            
            item = (jac_sim, testDocId)
            heapq.heappush(neighbors, item)
        i += 1
    
    while (i < len(docId)):
        if (docId[i] != doc):
            testDocId = docId[i]
            testDoc = documents[i]
            
            intersect = len((testDoc).intersection(check_doc))
            union = len((testDoc).union(check_doc))
            jac_sim = float(intersect)/union
            
            item = (jac_sim, testDocId)
            heapq.heappushpop(neighbors, item)
        i += 1

    '''
    Compute average jacc for the knn
    '''
    total_jac = 0
    knn = []
    for jacc in neighbors:
        knn.append(jacc[1])     #record the id of the document
        total_jac += jacc[0]

    avg_jac = float(total_jac)/(len(knn))
    ret = (avg_jac,knn)
    return ret





'''
Find average of average knn
'''
def avg_avg_knn(documents,docId):
    k = raw_input("How many neighbors would you like to find? ")

    while ((k.isdigit() == False) | (int(float(k)) >= (len(docId))) | (int(float(k)) <= 0)):
        print("Inappropriate k value chosen. Please enter appropriate integer value.")
        k = raw_input("How many neighbors would you like to find?")
    k = int(float(k))

    totalAvg = 0
    for i in range(len(docId)):
        cur = bruteForce(documents,docId,docId[i],k)
        totalAvg = totalAvg + cur[0]
        #print "The avg knn similarity for doc " + str(docId[i]) + " is " + str(cur[0])
        if i%100 == 0:   ###check if running
            print i

    avg_avg = float(totalAvg)/len(docId)
    print "###############################################################"
    print "###The average knn similarity across ALL documents is " + str(avg_avg)








'''
Function to compute hash function
Param: x - original 'row' number in char
       a - random number from 0 to 1-x and is prime
       b - random number from 0 to 1-x
       w - number of unique words in the selected documents
'''
def minHash(x,a,b,w):
    newIndex = (a * x + b) % w
    # print "x= " + str(x)
    # print "a= " + str(a)
    # print "b= " + str(b)
    # print "w= " + str(w)
    #print "newIndex = (a * x + b) % w: " + str(newIndex)
    return int(float(newIndex))



def getSigMatrix(documents):
    np.set_printoptions(suppress=True)
    print "----------------------- Generating Signature Matrix ------------------------"
    numWords = 0
    
    #generate a set of all wordIds in all documents
    # for i in range(len(documents)):
    #     allWords = allWords.union(documents[i])
    allWords = set.union(*documents)
    allWords = list(allWords)   #index of the wordId in list allWords is the original row number

    #####################Do we sort the words????#####################################################

    #allWords = sorted(allWords)
    
    ################################################################################################

    #print "all words: " + str(allWords)
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

    # for num in range(3,w,2):
    #     if all(num%i!=0 for i in range(2,int(math.sqrt(num))+1)):
    #         if (w%num != 0):
    #             a.append(num)
    #     if (len(a) >= 11):
    #         break
    #print "list of primes a: " + str(a)
    np.random.seed(411)
    b = list(np.random.choice(range(w),numRows, replace = False))
    
    # for j in range (numRows):
    #     print str(a[j%len(a)]) + " + " + str(b[j])
    #Generate the jth hash functions by using a[j%len(a)] and b[j]

    '''
    Generate signature matrix
    '''
    
    #Initialize a signature matrix with infinity in all slots
    sigMatrix = np.empty((numRows, len(documents),))
    sigMatrix[:] = float("inf")
    print sigMatrix

    wordcount = 0
    hashVal = [0] * numRows
    
    #update sig matrix
    for x in range (len(allWords)):          #for every row in character matrix
        wordcount+=1
        
        #start_time = timeit.default_timer() #start timer

        for j in range(numRows):   #compute hash values for row 'x' for each hash fcn and store in list hashVal
            hashVal[j] = minHash(x,a[j%len(a)],b[j],w)
        #print("Compute hashvalue time for each word: " + str(timeit.default_timer() - start_time))


        for doc in range(len(documents)):      #for every document, check if the word is in its set of words.
            if (allWords[x] in documents[doc]):
                
                #start_time = timeit.default_timer()
                for y in range(numRows):
                    
                    if (hashVal[y] < sigMatrix[y][doc]):     #y^th row, doc^th column
                        sigMatrix[y][doc] = hashVal[y]

        #print("Compute update time for each row: " + str(timeit.default_timer() - start_time))

        #print sigMatrix

    print "#words = " + str(wordcount)
    print sigMatrix
    return sigMatrix


def estJaccard(sigMatrix, docId, *argv):
    print "----------------------- Estimating Jaccard Similarity ------------------------"
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

    union = len(sigMatrix)
    intersection = 0
    for row in sigMatrix:
        if row[doc1_index] == row[doc2_index]:
            intersection += 1

    jac_est = float(intersection)/union

    print "Estimated Jaccard similarity of the two docs is " + str(jac_est)
    return jac_est






def LSH_buckets(sigMatrix, docId, numRows):
    r = raw_input("Enter the number of rows in each band:")
    while ((r.isdigit() == False) | (int(float(r)) <= 0) | (int(float(r)) > numRows) | (numRows % (int(float(r))) != 0 ) ):
        print "Inappropriate number of rows. Please re-enter."
        print "Make sure r divides " + str(numRows)
        r = raw_input("Enter the number of rows in each band:")
    
    r = int(float(r))
    b = int(float(numRows/r))
    
    list_of_dicts = []
    flipped_sigMatrix = sigMatrix.T
    start = 0
    end = r
    for times in range(b): #b times
        hashes = {} # tuples are unique keys, docIds are values
        for i in range(len(flipped_sigMatrix)): #look at each column
            bucket_key = []
            bucket = []
            for entry in range(start,end):
                bucket_key.append(flipped_sigMatrix[i][entry])
            bucket_key = tuple(bucket_key) #tuples have to be immutable
            if bucket_key in hashes.keys(): #if the band is already in the dictionary
                hashes[bucket_key].append(docId[i])
            else: #adds the key if it appears for the first time.
                bucket.append(docId[i])
                hashes[bucket_key] = bucket
        list_of_dicts.append(hashes)
        start += r
        end += r
    print list_of_dicts
    return list_of_dicts
        
    
def LSH(list_of_dicts, docId, sigMatrix, *argv):
    if argv: #if nonempty
        doc = argv[0]       #doc is the docId
        k = argv[1]         
    else:
        doc_to_examine = raw_input("Enter the ID of the document you wish to examine:")
        while ((doc_to_examine.isdigit == False) | (int(float(doc_to_examine)) <= 0) | (int(float(doc_to_examine)) not in docId)):
            print "Inappropriate document ID. Please re-enter."
            doc_to_examine = raw_input("Enter the ID of the document you wish to examine:")
        doc_to_examine = int(float(doc_to_examine))
    
        k = raw_input("Enter the number of nearest neighbors:")
        while ((k == False) | (int(float(k)) <= 0) | (int(float(k)) > len(docId))):
            print "Inappropriate k. Please re-enter."
            k = raw_input("Enter the number of nearest neighbors:")
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
                         
    #go through and keep a priority queue of the k biggest estimated JacSims
    while len(candidates) < k:
        np.random.seed(1229)
        toAdd = random.choice(docId)
        if toAdd != doc_to_examine:    
            candidates.add(toAdd) 
        print "randed " + str(candidates)

    candidates = list(candidates)
    k_neighbors = []
    heapq.heapify(k_neighbors)
    est_sim_count = 0
    average_est_sim = 0
    
    print "the candidates are: " + str(candidates)

    for cand in candidates[0:k]:
        est_jac = estJaccard(sigMatrix, docId, doc_to_examine, cand)
        node = (est_jac, cand)
        heapq.heappush(k_neighbors, node) #creates a priority queue with k elements. 
    print "aaa:" + str(k_neighbors)
    for cand2 in range(k,len(candidates)):
        est_jac = estJaccard(sigMatrix, docId, doc_to_examine, candidates[cand2])
        node2 = (est_jac, candidates[cand2])
        
        heapq.heappushpop(k_neighbors, node2)

    print k_neighbors
        
    for cur in k_neighbors:
        est_sim_count += cur[0]
    
    average_est_sim = float(est_sim_count)/k
    print average_est_sim
    return average_est_sim







def main():
    start_time = timeit.default_timer()
    data = readDocuments()           #data[0] is documents, data[1] is docId
    print("Read time: " + str(timeit.default_timer() - start_time))

    start_time = timeit.default_timer()
    sig = getSigMatrix(data[0])
    print("GetSignature time: " + str(timeit.default_timer() - start_time))

    list_of_dicts=LSH_buckets(sig, data[1], len(sig))
    LSH(list_of_dicts, data[1], sig)


    done = False

    while (done == False):
        print "-------Commands--------"
        print "To compute Actual Jaccard Similarity: Enter 1."
        print "To compute Estimated Jaccard Similarity: Enter 2."
        print "To compute brute force Similarity: Enter 3."
        print "To recompute signature matrix: Enter 4."
        print "To compute the avg of avg similarity of all docs: Enter 5."


        print "To quit, enter q."
        command = raw_input('Enter command: ')
        if (command == "1"):
            start_time = timeit.default_timer()
            actualJaccard(data[0],data[1])
            print("Compute actual Jaccard sim time: " + str(timeit.default_timer() - start_time))

        elif (command == "2"):
            start_time = timeit.default_timer()
            estJaccard(sig, data[1])
            print("Compute estimate Jaccard sim time: " + str(timeit.default_timer() - start_time))

        elif (command == "3"):
            print "--------------------- Using brute force to find the k nearest neighbors---------------------"
            start_time = timeit.default_timer()
            kneighbors = bruteForce(data[0],data[1])
            print "The Average Jaccard Similarity with k neighbors is: " + str(kneighbors[0])
            print "The k neighbors are " + str(kneighbors[1])
            print("Compute bruteforce time: " + str(timeit.default_timer() - start_time))

        elif (command == "4"):
            start_time = timeit.default_timer()
            sig = getSigMatrix(data[0])
            print("Compute getSigmatrix time: " + str(timeit.default_timer() - start_time))

        elif (command == "5"):
            start_time = timeit.default_timer()
            avg_avg_knn(data[0],data[1])
            print("Compute bruteforce avg_avg knn time: " + str(timeit.default_timer() - start_time))

        else:
            done = True





main()