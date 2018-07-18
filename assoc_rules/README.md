# Finding Association Rules

This program is written in python, and uses the apriori algorithm to identify the candidate itemsets to be used for finding association rules.

Two sets of rules are generates, one based on confidence and one based on interest.

## Getting Started

1. Unzip and put all files in assoc_rules.zip into the same directory. 
2. Place your input files under the same directory as the source codes:
    Name your data file as 'BobRoss.txt'
3. Download the required packages below.


### Prerequisites

- Ptyhon (2.7)
    a. NumPy (1.12.1)

## Usage

The program is in the file assoc_rules.py.
The commands to run each program is described below. 

### Command Line Parameters:

1. **min_sup (int)** - The minimum level of support needed for a frequent itemset.
2. **min_conf (float)** - The minimum level of confidence for a candidate association rule to be
   accepted.
3. **min_int (float)** - The minimum level of interest for a candidate association rule to be accepted.

#### Running assoc_rules.py

Sample commands:
```
python assoc_rules.py 25 0.9 0.7
```

## Design Decisions

(1) When  generating rules based on confidence, we used a layer based approach dependent on the size of the rhs of the rule.

(2) When generating rules based on interest, we used brute force. 

This is not the most efficient way to write this program, since we are using brute-force to generate all rules when based on interest. But for the purpose of this assignment, this program generates the two set of rules separately and independently.

## Known Bugs

None so far.