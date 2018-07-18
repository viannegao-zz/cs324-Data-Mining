'''
CS 324 Assignment 1 - kNN
This is a program that loops through all 300 images in the 
test_data.txt, and uses 900 images in the train_data.txt and 
the K Nearest Neighbor Algorithm to predict which class the image belongs to.
(all images are pre-processed 28x28 MNIST images of written digits 1, 2, 7).
Author: Vianne Gao
Date: April 5, 2017
'''

from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as sp
import sys
import random


'''
Function that computes the distance between each test data and every training data.
Param: matrix1 - training data in the form of an array of m vectors, m = # training data
	   matrix2 - test data loaded in the form of an array of n vectors, n = # test data
Output: a LIST of row vectors, one row vector for each test data, 
		each entry is a list of distance to each training data
'''
def getDistance (matrix1, matrix2, metric):
	#Euclidean Distance as metric
	if metric == "euclidean":
		distList = []
		for test_data in matrix2: #for each test data, loop through each of the m training data and compute distance
			sub_distList = []
			
			for train_data in matrix1:
				dist = np.linalg.norm(test_data - train_data)
				sub_distList.append(dist)
			
			distList.append(sub_distList)

	#Manhattan Distance as metric
	elif metric == "manhattan":
		distList = []
		
		for test_data in matrix2: #for each test data, loop through each train data and compute distance
			sub_distList = []

			for train_data in matrix1:
				dist = sp.distance.cityblock(test_data, train_data)
				sub_distList.append(dist)

			distList.append(sub_distList)
	
	else:
		print ("Wrong metric, enter euclidean or manhattan.")
		return
	
	return distList
	'''

	'''

'''
Function that takes the list of distance between one test data and all training data,
finds the labels of the k closest neighbors,
then output the majority vote to be the label of the test data.
Param: int k - number of neighbors
	   list dist - a list of distance between the one test data and all m training data
	   list label_train - a list of labels for all m training data
Output: A class label for the test data
'''
def kNearestNeighbors (k, dist, label_train):
		'''
		Find the k closest training data
		'''		
		dist, label_train = zip(*sorted(zip(dist, label_train)))    #sort distances for the test data
		kthDistance = dist[k-1] 									#distance of the kth closest neighbor

		#To avoid ties, count number of neighbors closer than or as close as the kth closest, and select all of them.
		#neighbors = label_train[:len([j for j in dist if j <= kthDistance])] 		
		neighbors = label_train[:k]


		'''
		Tally the votes for the current image
		'''
		votes = np.bincount(np.array(neighbors), minlength = 8)


		'''
		Look for ties in voting. If there is a tie, reduce k. 
		k will be at least 1.
		'''
		if ((np.count_nonzero(votes == (max(votes)))>1) & (k>1)):
			#print("There is a tie. Running kNN again with k = " + str(k-1) + ".")
			return kNearestNeighbors(int(k-1), dist, label_train) #recursion with k-1
		
		result = np.where(votes == (max(votes)))[0][0]
		return result

'''
Use kNearestNeighbor to classify all test data.
Param:  int k - number of neighbors to consider
		str metric - type of distance to use
		array set_train - array of m vectors representing m training data
Output: a list of class labels for all test data.
'''
def classify (k, metric, set_train):

	k = int(k)
	test = "test_data.txt" 				#select file for the test data
	lab_train = "train_labels.txt" 		#select file for the labels of the training data 
	
	'''
	load data into training matrix, train_label matrix, testing matrix, and test_label matrix.
	'''
	label_train = [int(i) for i in list(np.loadtxt(lab_train))]   # read train_labels.txt into a list
	set_test = np.loadtxt(test, delimiter=',') 					  # read test_data.txt into an array
	set_train = set_train										  # copy down the array or training data

	
	'''
	Use the getDistance function to generate a list of lists,
	one list for each test data,
	each list contains the distance of the test data to each training data
	'''
	distList = getDistance(set_train,set_test,metric)

	'''
	For each test data, use the kNearestNeighbors function to classify all test data 
	'''
	voteResult = []
	for i in range(len(distList)):
		#print("*************Running test on data " + str(i+1) + "*********************")
		curDist = distList[i] 											#select the ith test data
		voteResult.append(kNearestNeighbors (k, curDist, label_train))	#add result to voteResult


	return voteResult 

'''
Function that computes the accuracy and generates a confusion matrix
Param: list classification - a list of class labels assigned to each test data
Output: int acc - the accuracy of the classifier (# true positive / # test data )
'''

def getAccuracy(classification):

	yTe = np.loadtxt("test_labels.txt")
	y = classification
	acc = 1 - np.mean((np.array(y).flatten())!=yTe)
	#lab_test = "test_labels.txt" 								#select file for labels of the test data
	#label_test = [int(j) for j in list(np.loadtxt(lab_test))] 	#read test_label file into a list	

	# '''
	# True positive rate
	# '''
	# TP_1 = 0
	# TP_2 = 0
	# TP_7 = 0
	
	# '''
	# False positive rate broken down for each class, FP_ab means the number of class a classified as class b
	# '''
	# FP_12 = 0
	# FP_17 = 0
	# FP_21 = 0
	# FP_27 = 0
	# FP_71 = 0
	# FP_72 = 0

	# takeValueOne = 0
	# takeValueTwo = 0
	# correct = []
	# wrong = []

	# for i in range(len(classification)):
	# 	if (classification[i] == label_test[i]):
	# 		correct.append(i)
	# 		if (classification[i] == 1):
	# 			TP_1 += 1

	# 		elif (classification[i] == 2):
	# 			TP_2 += 1

	# 		elif (classification[i] == 7):
	# 			TP_7 += 1

	# 	if (classification[i] != label_test[i]):
	# 		wrong.append(i)
	# 		if (label_test[i] == 1):
	# 			if (classification[i] == 2):
	# 				FP_21 += 1
	# 			else:
	# 				FP_71 += 1
	# 		elif (label_test[i] == 2):
	# 			if (classification[i] == 1):
	# 				FP_12 += 1
	# 			else:
	# 				FP_72 += 1
	# 		elif (label_test[i] == 7):
	# 			if (classification[i] == 1):
	# 				FP_17 += 1
	# 			else:
	# 				FP_27 += 1
	
	# totalP = np.bincount(np.array(classification), minlength = 8)
	# totalActual = np.bincount(np.array(label_test), minlength = 8)
	
	
	# '''
	# False positive rate (classified positive - correct positive)
	# '''
	# FP_1 = totalP[1]-TP_1
	# FP_2 = totalP[2]-TP_2
	# FP_7 = totalP[7]-TP_7

	# '''
	# False Negative Rate (100 - TP)
	# '''
	# FN_1 = label_test.count(1) - TP_1
	# FN_2 = label_test.count(2) - TP_2
	# FN_7 = label_test.count(7) - TP_7
	# '''
	# True Negative (classified neg - false neg)
	# '''
	# totalNum = len(label_test)
	# TN_1 = totalNum - TP_1 - FP_1 - FN_1
	# TN_2 = totalNum - TP_2 - FP_2 - FN_2
	# TN_7 = totalNum - TP_7 - FP_7 - FN_7

	# '''
	# Plot a Confusion Matrix
	# '''
	# header = ["Actual","Class",""]
	# classes = ["Digit 1", "Digit 2", "Digit 7"]
	# data = np.array([[TP_1, FP_21, FP_71],
	# 				[FP_12, TP_2, FP_72],
	# 				[FP_17, FP_27, TP_7]])

	# row_format = "{:>12}" * (len(classes) + 2)
	
	# print ("--------------- Confusion Matrix ----------------")
	# print (row_format.format("","","","Predicted Class",""))
	# print (row_format.format("", "", *classes))
	# for aHeader, aClass, row in zip(header, classes, data):
	# 	print (row_format.format(aHeader, aClass, *row))

	# '''
	# Get Accuracy (proportion of true positive over all classifications)
	# '''
	# acc = float(TP_1 + TP_2 + TP_7)/totalNum
	return acc


'''
Function that displays a list of 784 gray-scale values as an image
Param: list data - a list of grey-scale values for a 28x28 image
'''
def display_image(data):
	data = np.array(data)
	data = np.reshape(data,(-1,28))
	plt.imshow(data)
	plt.show()


def main(k,metric):
	
	#Check if metric input is appropriate. If not, throw an error.
	if ((metric != "euclidean") & (metric != "manhattan")):
		print ("Wrong metric, enter euclidean or manhattan.")
		return
	
	#Read the training data file into an array
	train = "train_data.txt"   
	set_train = np.loadtxt(train,delimiter=',')
	
	#Check whether k is appropriate. If not, throw an error
	if ((k.isdigit() == False) | (int(float(k)) > set_train.shape[0]) | (int(float(k)) <= 0)):
		print("Inappropriate k value chosen. Please enter appropriate integer value.")
		return
	start_time = timeit.default_timer()

	#Use the function classify to classify all test data	
	classification = classify(k, metric, set_train)
	
	#Print the accuracy of the classifier and the confucion matrix onto the screen
	print ("-The Accuracy using k = " + str(k) + " and Metric = " + 
		str(metric) + " is: " + str(getAccuracy(classification)))
	print "Classification time: " + str(timeit.default_timer()-start_time)



main(sys.argv[1], sys.argv[2])




