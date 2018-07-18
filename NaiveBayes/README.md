# Writing Portfolio Classification with Naive Bayes

This program is written in python, and uses the Naive Bayes algorithm to classify and predict Carleton College writing porfolios as either need work or do not need work.

## Getting Started

1. Unzip and put all files in NB.zip into the same directory. 
2. Place your input files under the same directory as the source codes:
    Name your file "writingportfolio.csv"
3. Download the required packages below.


### Prerequisites

- Ptyhon (2.7)
    a. NumPy (1.12.1)
    b. SciPy (0.19.0)

## Usage

There is one program in this file: nb.py.

The commands to run each program is described below. 

Sample command to call:
```
python nb.py
```


## Design Decisions

#### Resolving ties in probability:
It is very unlikely, but if the likelihood of being classified as need work is the same as being classified as no need work, the classifier will choose to classify as need work.

#### Deciding ways to model different features:

The final set of features used are “Minnesota”, “international”, “Verbal SAT scores”, “Number of Submitted essays”, “Abroad Credits”, “AP Credits”, “CS Credits”, “English Credits”,  “Writing Credits” as discrete and “GPA” as continuous.

#### Removing Test Data

We removed the test data by:
1. (Discrete features) finding the count of the bin to which each value in the test data belong, and subtract the value by one to simulate the process of removing the test data. We can do this because the test data contribute 1 count to the bin to which they belong. This is also done to the total count of classes labeled 'need work' and 'no need work'.
2. (Continuous feature) remove the value of the test data from the list of all values from the continuous feature, recompute the mean and standard deviation and calculate probability using cdf.

#### Add One Smoothing

We used add one smoothing for all discrete feature calculations to avoid the possibility of getting data with feature values not present in the training data.

## Known Bugs

1. The quality of the input files are not taken into consideration by the programs. Thus, any missing data or discrepancy between the number of training datas and training labels will not be detected by the program. Thus, any unlabeled training data will not be used to classify the test data.
2. If there is unlabeled test data, then accuracy cannot be computed. An Index out of range error will be encountered.
3. Do not take into account the quality of training data. May have missing values and duplicates.
4. Datasize too small.

