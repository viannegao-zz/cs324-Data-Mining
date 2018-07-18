# Handwritten Digit Recognition with k-means

This program is written in python, and uses the k means algorithm to identify the number represented by 28-by-28 grayscale images, each containing a hand-written digit from 0 to 9.

## Getting Started

1. Unzip and put all files in kmeans.zip into the same directory. 
2. Place your input files under the same directory as the source codes:
    Name your training data as 'number_data.txt'; 
    Name yout training labels as 'number_labels.txt';
3. Download the required packages below.

### Prerequisites

- Ptyhon (2.7)
    a. NumPy (1.12.1)
    b. SciPy (0.19.0)
    c. Matplotlib (1.5.3)

## Usage

There are two programs in this file: kmeans.py and test.py. The same command line parameters are used for both programs.

The commands to run each program is described below. 

### Command Line Parameters:

**k** - an integer that is greater than 0 and less than the number of training data, indicating the number of clusters desired.

**init** - a string (rand, other or cheat), indicating the method of initialization.

#### Running kmeans.py

Sample commands:
```
python kmeans.py 10 random
```
#### Running test.py

This program has additional functionalities on top of the kmeans algorithm. 
1. Creates plots of the centroids found
2. uses the true labels to assess quality of clustering
3. Prints the SSE for each cluster
4. Allows 'cheating' initialization

Sample commands:
```
python other.py 10 cheat
```

## Design Decisions
#### Refined Centroid Initialization
The centroids are initialized using the Fayyad Refined Cluster Center algorithm. Sub-samples of size 500 (5% of the data) are sampled for 30 times from the original data points. Each sub-sample is clustered using the k-means algorithm with the initial centroids chosen randomly. After generating the centroids for each sample, we will cluster these centroids from 30 runs using the k means algorithm with the initial centroids chosen randomly. The final centroids from this process are the initial centroids to use when clustering the whole dataset.
This method is usually better than randomly choosing data points, because subsampling can provide information on the location of the true clusters. It reduces the number of iterations needed, thus reduces the run-time of the actual clustering process using k means. Also, this method helps to reduce the likelihood of being stuck in a local minima.
Reference: Bradley, Paul S., and Usama M. Fayyad. "Refining Initial Points for K-Means Clustering." ICML. Vol. 98. 1998.

#### Resolving ties in distance when assigning centroid:
For every data point, after calculating the distance to all k centroids, the data point is assigned to the centroid it is the closest to. In cases where there is a tie in distance, the first encountered centroid is chosen.

#### Determining convergence of clustering:
After every iteration, the percentage change in each centroid is computed. If the percentage change is 0 for every attribute (hence no changes happened in any centroid), then iteration stops and the converged centroids are returned.

#### Handling empty clusters:
After every assignment of data points to nearest clusters, replace any empty cluster's centroid with the datapoint that is the furthest away from its current centroid.

#### Computing change in centroid:
Using matrix algebra, the value of the current centroids are subtracted from the previous centroid. If the value of all entries are 0, then the clustering is considered to be converged.

## Known Bugs
None known yet.


