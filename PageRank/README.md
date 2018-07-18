# Protein Interactions using Page Rank Algorithm

This program is written in python, and uses the page rank algorithm to identify the proteins that highly interact with other proteins.

## Getting Started

1. Unzip and put all files in PageRank.zip into the same directory. 
2. Place your input files under the same directory as the source codes:
    Name your data file as 'wnt_edges.txt'
3. Download the required packages below.


### Prerequisites

- Ptyhon (2.7)
    a. NumPy (1.12.1)
    b. Matplotlib (1.5.3)

## Usage

The program is in the file pageRank.py.
The commands to run each program is described below. 

### Command Line Parameters:

**beta** - an integer that is between 0 and 1, representing the teleportation value to use.

**iters** - a positive integer indicating how many iterations to use for the power method.
#### Running pageRank.py

Sample commands:
```
python pageRank.py 0.9 50
```

## Design Decisions

#### Invalid beta:
If a beta value greater than 1 or less than 0 is given, we will print an error message and use beta = 1.

#### Invalid iterations:
If a iters value less than 0 is given, we will print an error message and use iters = 10.

## Known Bugs

None so far.