'''
CS 324 Assignment 1 - kNN
This is a program that uses 5-fold cross validation on the training data to test the performance of knn algorithm, by 
computing the confusion matrix and accuracy, using only the 900 images in the train_data.txt and 
the K Nearest Neighbor Algorithm to predict which class the image belongs to.
(all images are pre-processed 28x28 MNIST images of written digits 1, 2, 7).

Author: Vianne Gao
Date: 
'''

from io import StringIO
import numpy as np
import random
import scipy.spatial as sp
import sys

'''
Function that computes the distance between each test data and every training data.
Param: array matrix1 - training data in the form of an array of m vectors, m = # training data
	   array matrix2 - test data loaded in the form of an array of n vectors, n = # test data
	   str metric - which distance measurement to use
Output: a LIST of row vectors, one row vector for each test data, 
		each entry is a list of distance to each training data
'''
def getDistance (matrix1, matrix2, metric):
	#Euclidean Distance as metric
	if metric == "euclidean":
		distList = []
		
		for test_data in matrix2: #for each test data, loop through each training data and compute distance
			sub_distList = []
			
			for train_data in matrix1:
				dist = np.linalg.norm(test_data - train_data)
				sub_distList.append(dist)
			
			distList.append(sub_distList)

	#Manhattan Distance as metric
	else:
		distList = []
		
		for test_data in matrix2: #for each test data, loop through each train data and compute distance
			sub_distList = []

			for train_data in matrix1:
				dist = sp.distance.cityblock(test_data, train_data)
				sub_distList.append(dist)

			distList.append(sub_distList)
	
	return distList 


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
		dist, label_train = zip(*sorted(zip(dist, label_train))) #sort the distances
		#kthDistance = dist[k-1] #distance of the kth closest neighbor

		#To avoid ties, count number of neighbors closer than or as close as the kth closest, and select all of them.
		#neighbors = label_train[:len([j for j in dist if j <= kthDistance])] 		
		neighbors = label_train[:k]

		'''
		Tally the votes for the current image
		'''
		votes = np.bincount(np.array(neighbors), minlength = 8)

		'''
		Look for ties in voting. If there is a tie, reduce k. k will be at least 1.
		'''
		if ((np.count_nonzero(votes == (max(votes))) > 1) & (k > 1)):
			print("There is a tie. Running kNN again with k-1 = " + str(k - 1) + ".")
			return kNearestNeighbors(int(k - 1), dist, label_train)
		
		result = np.where(votes == (max(votes)))[0][0]
		return result

'''
Use kNearestNeighbor to classify all test data.
Param:  int k - number of neighbors to consider
		str metric - type of distance to use
		array set_train - array of m vectors representing m training data
		array set_test - array of n vectors representing n test data
		list label_train - list of class labels for the m training data
Output: a list of class labels for all test data.
'''
def classify (k, metric, set_train, set_test, label_train):
	
	k = int(float(k))

	label_train = label_train   # copy down list of labels training data
	set_test = set_test         # copy down an array of all test data
	set_train = set_train		# copy down an array of all training data

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
		dist = distList[i] 					#select the ith test data
		voteResult.append(kNearestNeighbors (k, dist, label_train))

	return voteResult 

'''
Function that computes the accuracy and generates a confusion matrix
Param: list classification - a list of class labels assigned to each test data
	   list label_test - a list of actual labels for the test data
Output: int acc - the accuracy of the classifier (# true positive / # test data )
'''
def getAccuracy(classification, label_test):

	label_test = label_test #copy down the list of labels for the test data
	
	'''
	True positive rate
	'''
	TP_1 = 0
	TP_2 = 0
	TP_7 = 0

	'''
	False positive rate broken down for each class, FP_ab means the number of class a classified as class b
	'''
	FP_12 = 0
	FP_17 = 0
	FP_21 = 0
	FP_27 = 0
	FP_71 = 0
	FP_72 = 0

	for i in range(len(classification)):
		if (classification[i] == label_test[i]):
			if (classification[i] == 1):
				TP_1 += 1
			elif (classification[i] == 2):
				TP_2 += 1
			elif (classification[i] == 7):
				TP_7 += 1
		if (classification[i] != label_test[i]):
			if (label_test[i] == 1):
				if (classification[i] == 2):
					FP_21 += 1
				else:
					FP_71 += 1
			elif (label_test[i] == 2):
				if (classification[i] == 1):
					FP_12 += 1
				else:
					FP_72 += 1
			elif (label_test[i] == 7):
				if (classification[i] == 1):
					FP_17 += 1
				else:
					FP_27 += 1
	
	totalP = np.bincount(np.array(classification), minlength = 8)
	totalActual = np.bincount(np.array(label_test), minlength = 8)
	
	'''
	False positive rate added up (classified positive - correct positive)
	'''
	FP_1 = totalP[1]-TP_1
	FP_2 = totalP[2]-TP_2
	FP_7 = totalP[7]-TP_7

	'''
	False Negative Rate (100 - TP)
	'''
	FN_1 = label_test.count(1) - TP_1
	FN_2 = label_test.count(2) - TP_2
	FN_7 = label_test.count(7) - TP_7
	
	'''
	True Negative (classified neg - false neg)
	'''
	totalNum = len(label_test)
	TN_1 = totalNum - TP_1 - FP_1 - FN_1
	TN_2 = totalNum - TP_2 - FP_2 - FN_2
	TN_7 = totalNum - TP_7 - FP_7 - FN_7

	'''
	Plot a Confusion Matrix
	'''
	header = ["Actual","Class",""]
	classes = ["Digit 1", "Digit 2", "Digit 7"]
	data = np.array([[TP_1, FP_21, FP_71],
					[FP_12, TP_2, FP_72],
					[FP_17, FP_27, TP_7]])

	row_format = "{:>12}" * (len(classes) + 2)
	
	print ("--------------- Confusion Matrix ----------------")
	print (row_format.format("","","","Predicted Class",""))
	print (row_format.format("", "", *classes))
	for aHeader, aClass, row in zip(header, classes, data):
		print (row_format.format(aHeader, aClass, *row))

	'''
	Get Accuracy (proportion of true positive over all classifications)
	'''
	acc = float(TP_1 + TP_2 + TP_7)/totalNum
	return acc


'''
Function to perform a 5-fold cross validation on the training data.
Param: int k - k value
	   str metric - type of distance to use
Output: Accuracy of the current classifier given k and metric
'''
def fiveCV(k,metric):
	
	'''
	load training data
	'''
	set_train = np.loadtxt("train_data.txt",delimiter=',')
	label_train = [int(i) for i in list(np.loadtxt("train_labels.txt"))] #list of labels for each train image
	
	'''
	Divide data into 5 subsets
	'''
	#data_label = list(zip(set_train,label_train))
	#random.shuffle(data_label)
	#set_train,label_train = zip(*data_label)
	
	set_train = list(set_train)
	label_train = list(label_train)
	s = int(round(float(len(set_train)) / 5)) # size of each subset
	
	#Check whether k is appropriate. If not, throw and error.
	if ((k.isdigit() == False) | (int(float(k)) > (len(set_train))-s) | (int(float(k)) <= 0)):
		print("Inappropriate k value chosen. Please enter appropriate integer value.")
		return
	
	'''
	Run classification on the 5 folds and compute accuracy
	'''
	order = list(range(len(set_train)))				#generate a list representing the order of training data

	perm_label = label_train[:] 					#copy labels
	order_label = list(zip(order, perm_label))		
	random.shuffle(order_label)
	order, perm_label = zip(*order_label)			#generate a permuted index and label of training data
	
	order = list(order)
	perm_label = list(perm_label)

	classification = []

	for i in range (5):
		start = i*s
		end = i*s+s
		if (i == 4):
			#end = len(set_train)
			end = len(order)
		
		'''
		generate test subset (data and label)
		'''
		#sub_test_data = set_train[start : end]
		#sub_test_label = label_train[start : end]
		
		sub_test_data = []
		sub_test_label = []
		sub_test_index = order[start:end]
		
		for p in range (len(sub_test_index)):
			sub_test_data.append(set_train[sub_test_index[p]])
			sub_test_label.append(label_train[sub_test_index[p]])
		
		'''
		generate training subset (data and label)
		'''
		# sub_train_data = set_train[:] #make copy
		# sub_train_label = label_train[:]
		# del sub_train_data[start : end]
		# del sub_train_label[start : end]
		
		sub_train_index = order[:]
		sub_train_data = []
		sub_train_label = []
		del sub_train_index[start:end]

		for q in range (len(sub_train_index)):
			sub_train_data.append(set_train[sub_train_index[q]])
			sub_train_label.append(label_train[sub_train_index[q]])

		#classify(k, metric, set_train, set_test, label_train)
		classified = classify(k, metric, sub_train_data, sub_test_data, sub_train_label)
		classification.append(classified)
	
	classification = sum(classification, [])

	#print ("Classification: " + str(classification))
	accuracy = getAccuracy(classification, perm_label)
	print ("-The Accuracy using k = " + str(k) + " and Metric = " + 
		str(metric) + " is: " + str(accuracy))
	return accuracy
	

def main(k,metric):
	#Check metric is appropriate
	if ((metric != "euclidean") & (metric != "manhattan")):
		print ("Wrong metric, enter euclidean or manhattan.")
		return

	return fiveCV(k,metric)
	
if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2])



