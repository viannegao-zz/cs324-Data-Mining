'''
Data Mining Assignment 5 - Association Rules
Author: Vianne Gao
'''
import sys
from itertools import chain, combinations
import itertools
from collections import defaultdict, Counter
import sys
import numpy as np
import heapq


'''
Function to read and preprocess data from txt file.
Param: str file - name of the file
Return: list allTrans: list of all transactions, each represented by 
                        a list of items it contains
        dict doc_with_item: items stored as keys and the value is a 
                        list of transactions containing the item
        list all_items: a list of the actual name of the items
        int numTrans: Number of transactions used

'''
def readData(file):
    print "------------------------- Reading Text Documents from File ------------------------"
    data = np.genfromtxt (file, delimiter=",", skip_header = 1) # Read all transactions from file
    with open('BobRoss.txt', 'r') as f: # Read the title row to get the name of items
        all_items = [word for word in f.readline().split(',')]
    feature = data.T  # Transpose the matrix so that rows are items
    transId = feature[0] 
    numTrans = len(transId)

    # Get a dictionary of items as keys and list of transactions containing the item as values
    doc_with_item = dict()
    for curItem in range (4,len(feature)): #update value for each item
        doc_with_item[curItem] = np.where(feature[curItem] == 1)[0].tolist()

    # Get a list of all transactions as a list of the items it contains
    data = np.delete(data,[0,1,2,3],axis=1)
    allTrans = []
    for i in range (len(data)):
        allTrans.append(np.where(data[i] == 1)[0].tolist())

    return allTrans, doc_with_item, all_items,numTrans

'''
Function to compute the support of distinct items in the transactions
and find frequent item sets of size 1.
Param: list allTrans: list of all transactions, each represented by 
                        a list of items it contains
       int min_sup: minimum support needed to be frequent
       int numTrans: Number of transactions used
Return: list candidate: a list of all frequent items of size 1
'''
def countDistinct(allTrans, min_sup, all_items):
    counter = Counter() # Create a counter to count the occurence of each item
    for trans in allTrans:
        counter.update(trans)
    candidate = list(key for key in counter if counter[key] >= min_sup)
    # Print the frequent items of size 1 and their support to screen.
    for key in candidate:
        print str(all_items[key+4]) + " : " + str(counter[key])
    return candidate

'''
Function to generate candidates of size k based on candidates of size k-1, for k > 1
Param: list freq_size1: a list of frequent item set of size 1
       int k: the size of the frequent item set we want
Return: list newL: a list of frequent item set of size k
'''
def generateCandidates(freq_size1,L,k):
    # to generate size 2 frequent items, produce all possible combinations from freq_size1
    if k == 2:
        newL = list(itertools.combinations(freq_size1,k))
        newL = [tuple(sorted(cand)) for cand in newL] #sort the frequent item sets
    # to generate size k > 2 frequent items, combine two frequent item set of size k-1
    #   if the first k-2 items are the same.
    else:
        newL = []
        for i in range(len(L)-1):
            for j in range(i+1,len(L)):
                if L[i][:-1] == L[j][:-1] and L[i] != L[j]:
                    newL.append(tuple(sorted(tuple(set(L[i]).union(set(L[j]))))))
    return newL

'''
Function to use the apriori algorithm to generate all frequent item sets
Param: int min_sup: support threshold we are interested in
       list allTrans: a list of all transactions
       list all_items: a list of the actual names of the items
       int numTrans: number of transactions
'''
def apriori(min_sup, allTrans, doc_with_item, all_items, numTrans):
    freq_size1 = countDistinct(allTrans,min_sup,all_items)
    print "Number of frequent sets of size 1: " + str(len(freq_size1))
    L = list(tuple([item]) for item in freq_size1) #get list of frequent item of size 1
    k = 2 # k = frequent item set size we currently want
    allCand = []
    while L != []: #while  L_k is not 0, find L_k+1
        allCand.append(set(L))
        L = generateCandidates(freq_size1,L,k)
        tempL = []
        for cand in L: #check if each candidate item set is frequent
            setList = []
            for i in cand: #for each item in the cand, find all docs that contain it
                setList.append(set(doc_with_item[i+4]))
            #find the support for the current candidate and check if it is frequent
            curSup = len(set.intersection(*setList)) #look at how many trans contain the item
            if curSup >= min_sup: #if frequent, the candidate is a frequent item
                tempL.append(cand)
            L = tempL
        print "Number of frequent sets of size " + str(k) + " : "+ str(len(L))

        k += 1
    return allCand

'''
Function to get the support of an itemset
Param: tuple itemset: a candidate itemset to check support for
       dict doc_with_item: keeping track of which documents contain each item
Return: Support of the itemset, which is the number of documents containing it
'''
def getSup(itemset,doc_with_item):
    setList = []
    for item in itemset:
        setList.append(set(doc_with_item[item+4]))
    return len(set.intersection(*setList))

'''
Function to compute the confidence and interest of a rule
Param: tuple lhs: LHS of the rule
       tuple rhs: RHS of the rule
       dict doc_with_item
       int numTrans
'''
def getConf_Int(lhs,rhs,doc_with_item,numTrans):
    num = getSup(tuple(set(lhs).union(set(rhs))),doc_with_item)
    denom = getSup(lhs,doc_with_item)
    conf = float(num)/denom
    interest = abs(conf - (getSup(rhs,doc_with_item)/float(numTrans)))
    return conf,interest

'''
Function to get rules based on confidence
Used the apriori principle to generate rhs itemsets
Return: all rules with at least confidence above min_conf
'''
def getRules_conf(min_conf, allCand, numTrans,doc_with_item):
    allLHS = []
    allRHS = []
    for f_size in range(1,len(allCand)): #for each set of frequent item set of size f_size
        for itemset in range(len(allCand[f_size])): #for each itemset
            #use layer method, generate rules by incrementing the size of rhs
            possibleRHS = list(list(allCand[f_size])[itemset]) #at first, any item can be rhs
            j_size = 1 #j_size tracks the size of rhs

            while(j_size <= len(list(allCand[f_size])[itemset])-1): 
                candRHS = []
                if j_size == 1: #if rhs is size 1, generate rhs with size 2 using combinations
                    curRHS = list(itertools.combinations(possibleRHS,j_size))
                    curRHS = [tuple(sorted(cand)) for cand in curRHS]
                else: #if rhs has size k > 1, use the apriori() to generate rhs of size k+1
                    curRHS = possibleRHS
                #Generate rules
                for rhs in curRHS: #for each rhs, generate lhs by using set difference
                    lhs = set(list(allCand[f_size])[itemset]).difference(set(rhs))
                    conf, interest = getConf_Int(lhs,rhs,doc_with_item,numTrans)
                    if conf >= min_conf:
                        allRHS.append(rhs)
                        allLHS.append(lhs)
                        candRHS.append(rhs)
                if j_size == 1:
                    size1_cand = [item[0] for item in candRHS]
                j_size += 1
                #generate the next layer of possible rhs itemsets
                possibleRHS = generateCandidates(size1_cand,candRHS,j_size)

    return allLHS, allRHS

'''
Function to get rules based on interest (Brute Force)
Return: all rules with at least confidence above min_int
'''
def getRules_int(min_int,allCand, numTrans,doc_with_item):
    allLHS = []
    allRHS = []
    for f_size in range(1,len(allCand)): #repeat for each set of frequent item set of size f_size
        for itemset in range(len(allCand[f_size])): #repeat for each itemset in current set of frequent item
            possibleRHS = list(list(allCand[f_size])[itemset])
            j_size = 1 #track size of rhs

            while(j_size <= len(list(allCand[f_size])[itemset])-1):
                #generate all combination of size j_size from size 1 frequent items
                curRHS = list(itertools.combinations(possibleRHS,j_size))
                curRHS = [tuple(sorted(cand)) for cand in curRHS]
                for rhs in curRHS: #for each rhs, generate lhs by using set difference
                    lhs = set(list(allCand[f_size])[itemset]).difference(set(rhs))
                    conf, interest = getConf_Int(lhs,rhs,doc_with_item,numTrans)
                    if interest >= min_int:
                        allRHS.append(rhs)
                        allLHS.append(lhs)
                j_size += 1
    return allLHS, allRHS

'''
Function to rank the rules based on confidence
Print: the top 10 rules and their confidence score
'''
def rankRules_conf(allLHS,allRHS,all_items,doc_with_item,numTrans):
    conf_list = []
    print "The number of rules found is: " + str(len(allRHS))
    if len(allRHS) == 0:
        return
    #Rank the rules
    for ind in range(len(allLHS)):
        conf, interest = getConf_Int(allLHS[ind],allRHS[ind],doc_with_item,numTrans)
        conf_list.append(conf)
    #Use heap to find the 10 rules with highest score
    ruleRank = []
    heapq.heapify(ruleRank)
    i = 0
    while i < 10 and i < len(allLHS):
        conf, interest = getConf_Int(allLHS[i],allRHS[i],doc_with_item,numTrans)
        item = (conf,i)
        heapq.heappush(ruleRank,item)
        i += 1
    while i < len(allLHS):
        conf, interest = getConf_Int(allLHS[i],allRHS[i],doc_with_item,numTrans)
        item = (conf,i)
        heapq.heappushpop(ruleRank,item)
        i += 1
    #Print the 10 best scores based on confidence
    for j in xrange (len(ruleRank),0,-1):
        left = []
        right = []
        for item1 in allLHS[ruleRank[j-1][1]]:
            left.append(all_items[item1+4])
        for item2 in allRHS[ruleRank[j-1][1]]:
            right.append(all_items[item2+4])
        right = ', '.join(rword for rword in right)
        left = ', '.join(lword for lword in left)
        print " "
        print "Rule: If " + str(left) + ", then " + str(right)
        print "Score for rule: " + str(ruleRank[j-1][0])

'''
Function to rank the rules based on interest
Print: the top 10 rules and their interest score
'''
def rankRules_int(allLHS,allRHS,all_items,doc_with_item,numTrans):
    int_list = []
    print "The number of rules found is: " + str(len(allRHS))
    if len(allRHS) == 0:
        return
    #Rank all rules based on interest
    for ind in range(len(allLHS)):
        conf, interest = getConf_Int(allLHS[ind],allRHS[ind],doc_with_item,numTrans)
        int_list.append(interest)
    # Use a heap to find the 10 rules with the highest score
    ruleRank = []
    heapq.heapify(ruleRank)
    i = 0
    while i < 10 and i < len(allLHS):
        conf, interest = getConf_Int(allLHS[i],allRHS[i],doc_with_item,numTrans)
        item = (interest,i)
        heapq.heappush(ruleRank,item)
        i += 1
    while i < len(allLHS):
        conf, interest = getConf_Int(allLHS[i],allRHS[i],doc_with_item,numTrans)
        item = (interest,i)
        heapq.heappushpop(ruleRank,item)
        i += 1
    #Print the 10 best scores based on interest
    for j in xrange (len(ruleRank),0,-1):
        left = []
        right = []
        for item1 in allLHS[ruleRank[j-1][1]]:
            left.append(all_items[item1+4])
        for item2 in allRHS[ruleRank[j-1][1]]:
            right.append(all_items[item2+4])
        right = ', '.join(rword for rword in right)
        left = ', '.join(lword for lword in left)
        print " "
        print "Rule: If " + str(left) + ", then " + str(right)
        print "Score for rule: " + str(ruleRank[j-1][0])
'''
Generate two sets of rules (1) based on confidence and (2) based on interest
'''
def main(min_sup,min_conf,min_int):
    allTrans, doc_with_item, all_items, numTrans = readData('BobRoss.txt')
    allCand = apriori(int(min_sup),allTrans, doc_with_item, all_items, numTrans)
    allLHS, allRHS = getRules_conf(float(min_conf),allCand,numTrans,doc_with_item)
    intLHS, intRHS = getRules_int(float(min_int),allCand,numTrans,doc_with_item)
    print " "
    print "---------------(1) Rules based on Confidence-------------- "
    rankRules_conf(allLHS, allRHS, all_items,doc_with_item,numTrans)
    print " "
    print "---------------(2) Rules based on Interest---------------- "
    rankRules_int(intLHS, intRHS, all_items, doc_with_item, numTrans)


main(sys.argv[1],sys.argv[2],sys.argv[3])