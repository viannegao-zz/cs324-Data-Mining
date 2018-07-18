'''
This code runs the knn-cv.py for both metrics and k in {1,3,5,7,9}.
'''
import matplotlib.pyplot as plt
from knn_cv import *
import sys

def runTest():
	k = [1,3,5,7,9]
	metric = ["euclidean", "manhattan"]
	accuracy = []
	for i in range (len(metric)):
		for j in range (len(k)):
			acc = fiveCV(str(k[j]), str(metric[i]))
			accuracy.append(acc)
	return accuracy




def main(repeats):
	totalAcc = [0,0,0,0,0,0,0,0,0,0]
	repeats = int(float(repeats))
	for i in range (repeats):
		accuracy = runTest()
		for j in range (10):
			totalAcc[j]  += accuracy[j]
	

	for m in range (10):
		totalAcc[m] = float(totalAcc[m])/repeats
	
	euc = totalAcc[:5]
	man = totalAcc[5:]

	k = [1,3,5,7,9]
	metric = ["euclidean", "manhattan"]

	plt.plot(k, euc, marker = 'o', linestyle = '--', color = 'r', label = 'Euclideam')
	plt.plot(k, man, marker = '^', linestyle = '--', color = 'b', label = 'Manhattan')
	plt.xlabel("k value")
	plt.ylabel("Accuracy = True Positive/Total Classifications")
	plt.title("5-fold Cross Validation Result")
	plt.legend()
	plt.show()
main(sys.argv[1])
