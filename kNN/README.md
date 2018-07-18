# Handwritten Digit Recognition with k-Nearest Neighbors

This program is written in python, and uses the k Nearest Neighbors algorithm to identify the number represented by 28-by-28 grayscale images, each containing a hand-written digit that is either 1, 2 or 7.

## Getting Started

1. Unzip and put all files in knn.zip into the same directory. 
2. Place your input files under the same directory as the source codes:
    Name your training data as 'train_data.txt'; 
    Name yout training labels as 'train_labels.txt';
    Name your test data as 'test_data.txt';
    Name your test labels as 'test_labels.txt';
3. Download the required packages below.


### Prerequisites

- Ptyhon (2.7)
    a. NumPy (1.12.1)
    b. SciPy (0.19.0)
    c. Matplotlib (1.5.3)

## Usage

There are two programs in this file: knn.py and knn-cv.py. The same command line parameters are used for both programs.

The commands to run each program is described below. 

### Command Line Parameters:

**k** - an integer that is greater than 0 and less than the number of training data, indicating the number of closest neighbors to consider during classification.

**metric** - a string (euclidean or manhattan), indicating the metric used to calculate the difference between two images.
#### Running knn.py

This program loops through all images in the test data, and uses the K nearest neighbors algorithm to predict their classes based on the images in the training data. 

Sample commands:
```
python knn.py 3 euclidean
python knn.py 1 manhattan
```
#### Running knn-cv.py

This program uses 5-fold cross validation on the training data, and computes a confusion matrix and accuracy for the classifier on the training data.

Sample commands:
```
python knn-cv.py 3 euclidean
python knn-cv.py 1 manhattan
```

## Design Decisions

#### Resolving ties in distance:

After computing the distance from one test data to each training data, the k closest training data are chosen to classify the test data. However, if more than k training data are as close as the kth closest one, then only the first k encountered training data are selected. This decision is based on the fact that the number of features for each data is relatively large, so the chance of getting a tie in distance is relatively small. 

Indeed, comparing this method of resolving ties to an alternative method, which includes all training data as close as the kth closest one, the accuracy of the classifier was not affected when testing the classifier with the test data and training data for CS324 Assignment 1.
The disadvantages of using this method is that when we are given a dataset with less features, or datasets where the datapoints are difficult to distinguish, then this method may result in bias in classification. 

#### Resolving ties in votes:

After the k nearest neighbors have been chosen for a test data, the majority label for the k neighbors determines the label of the test data. However, if there is more than one label receives the majority vote, then the tie is resolved by running the k nearest neighbors algorithm again with k-1. This process repeats until there are no ties. Observe that k is at least 1, since there are no ties when k = 1.

The rationale behind the decision to decrease k when a tie is reached is as follows. Firstly, when the classifier is tested on the test data and training data for CS324 Assignment 1, it is observed that the classifier has a higher accuracy when k is smaller. Secondly, decreasing k guarantees that the tie will eventually be resolved when k=1.

#### Partitioning the training data

In the program knn-cv.py, to perform a 5-fold cross validation, the training data is first permuted and then partitioned, because the original training data might be sorted based on labels (this is bad). This is done by generating a random list of indices corresponding to the order in which the respective rows are rearranged in the permuted matrix.
Thus, the accuracy of the classifier might vary slightly each time knn-cv.py is ran, because the training data is partitioned randomly each time.

#### Confusion matrix and accuracy

In the program knn-cv.py, the confusion matrix and accuracy printed on screen at the end represents the overall result of the 5-fold cross validation.

## Known Bugs

1. The quality of the input files are not taken into consideration by the programs. Thus, any missing data or discrepancy between the number of training datas and training labels will not be detected by the program. Thus, any unlabeled training data will not be used to classify the test data.
2. If there is unlabeled test data, then accuracy cannot be computed. An Index out of range error will be encountered.

