'''
Data Mining HW3 - Naive Bayes Classifier
Author: Vianne Gao
Date: April 19 2017
To adjust input file, change file name in line 40.
'''
import numpy as np
import scipy
from scipy import stats

'''
Structure of classifier:
1. Read in all the data from a csv file. Store the data for each portfolio in matrices.
2. Define the categories in each feature we chose to treat as discrete and COUNT the number of occurence of 
    each category given it needs work or do not need work.
3. Compute the probability of being in a certain category given a portfolio need work or no need work
    Used add one smoothing and log probabilities.
4. Compute continuous probability given the mean and sd of training data
5. Compare the probability of being in each class and classify the porftolio as the larger probability one.
6. Print the accuracy of the classifier and the portfolios classified incorrectly
'''



'''
Function to preprocess data in the form of a csv file.
Compute the components necessary for calculating probabilities.
Return: list preprocessed - components used to calculate probability
'''
def preprocess():
    '''
    Read data from csv file, store each project as a row in a matrix. 
    Matrix data:
    For the given test data, there are 141 rows, 15 columns
    (0_Project ID; 1_Minnesota; 2_International; 3_BirthYear;
     4_VerbalSAT; 5_MathSAT; 6_GPA; 7_#essays; 8_AbroadCredits;
     9_AP; 10_CS; 11_English; 12_Science; 13_Writing; 14_NeedsWork)
    Transpose the matrix to get arrays of each feature.
    '''
    data = np.genfromtxt ('writingportfolio.csv', delimiter=",",
        skip_header = 1,skip_footer = 34)
    feature = data.T

    '''
    Create two matrices: one for portfolios that Need work and one for No Need work
    '''
    needWork = data[data[:,-1]==1]   #matrix of all portfolios that needs work
    needWorkFeat = needWork.T

    noWork = data[data[:,-1]==-1]    #matrix of all portfolios that do not need work
    noWorkFeat = noWork.T


    '''
    Define the bins in each discrete feature
    '''
    cat_mn = ['x == 0', 'x == 1']
    cat_int = ['x == 0', 'x == 1']
    cat_year = ['x < 1982', 'x == 1982', 'x == 1983', 'x == 1984', 'x == 1985', 'x > 1985']
    cat_essay = ['x < 3', 'x == 3', 'x == 4', 'x == 5', 'x > 5']
    cat_abroad = ['x < 11', '11<= x <= 20', '21 <= x <= 30', '31 <= x <= 40', 'x>40']
    cat_ap = ['x <= 5', '6 <= x <= 11', '12 <= x <= 18', '19 <= x <= 24', '25 <= x <= 30',
                '31 <= x <= 36', 'x > 36']
    cat_sat = ['x < 300', '300 <= x < 400', '400 <= x < 500', '500 <= x < 600', '600 <= x < 700',
                    '700 <= x < 800', 'x >= 800']
    cat_cs = ['x <= 5', '6 <= x <= 12', '13 <= x <= 20', '21 <= x <= 40', 'x > 40']
    cat_eng = ['x <= 5', '6 <= x <= 12', '13 <= x <= 20', '21 <= x <= 40', 'x > 40']
    cat_sci = ['x <= 5', '6 <= x <= 11', '12 <= x <= 20', '21 <= x <= 30', '31 <= x <= 40', 
                '41 <= x <= 60', 'x > 60']



    ###########################################################################
    # Find the components needed to compute 'probability'                     #
    # Count instances for each feature in each label (need / no need work)    #
    ###########################################################################
    
    countNeedWork = len(needWork)       #Count total number of cases labeled need work
    countNoWork = len(noWork)           #Count total number of cases labeled no need work

    ### Given Need Work (represented as <feature>_1 for <feature> in each column) ###

    # Minnesota [0,1]
    countMn_1 = [np.count_nonzero(needWorkFeat[1] == 0), np.count_nonzero(needWorkFeat[1] == 1)]

    # International [0,1]
    countInt_1 = [np.count_nonzero(needWorkFeat[2] == 0), np.count_nonzero(needWorkFeat[2] == 1)]

    # Birth Year [#<1982, #=1982, #=1983, #=1984, #=1985, #>1985]
    countYear_1 = [np.count_nonzero(needWorkFeat[3] < 1982),np.count_nonzero(needWorkFeat[3] == 1982),
        np.count_nonzero(needWorkFeat[3] == 1983),np.count_nonzero(needWorkFeat[3] == 1984),
        np.count_nonzero(needWorkFeat[3] == 1985),np.count_nonzero(needWorkFeat[3] > 1985)]

    # Verbal SAT [mean, sd]
        # Gaussian Model
    allVerbal_1 = needWorkFeat[4][:] #This list stores all the feature values in order to calculate mean and sd (if we choose to use Gaussian model)
        # Categorical
    countVerbal_1 = [np.count_nonzero(needWorkFeat[4] < 300), 
                        np.count_nonzero((needWorkFeat[4] >= 300) & (needWorkFeat[4] < 400)),
                        np.count_nonzero((needWorkFeat[4] >= 400) & (needWorkFeat[4] < 500)), 
                        np.count_nonzero((needWorkFeat[4] >= 500) & (needWorkFeat[4] < 600)),
                        np.count_nonzero((needWorkFeat[4] >= 600) & (needWorkFeat[4] < 700)),
                        np.count_nonzero((needWorkFeat[4] >= 700) & (needWorkFeat[4] < 800)),
                        np.count_nonzero(needWorkFeat[4] >= 800)]

    # Math SAT [mean, sd] 
        #Gaussian Model
    allMath_1 = needWorkFeat[5][:] #This list stores all the feature values in order to calculate mean and sd (if we choose to use Gaussian model)
        #Categorical
    countMath_1 = [np.count_nonzero(needWorkFeat[5] < 300), 
                        np.count_nonzero((needWorkFeat[5] >= 300) & (needWorkFeat[5] < 400)),
                        np.count_nonzero((needWorkFeat[5] >= 400) & (needWorkFeat[5] < 500)), 
                        np.count_nonzero((needWorkFeat[5] >= 500) & (needWorkFeat[5] < 600)),
                        np.count_nonzero((needWorkFeat[5] >= 600) & (needWorkFeat[5] < 700)),
                        np.count_nonzero((needWorkFeat[5] >= 700) & (needWorkFeat[5] < 800)),
                        np.count_nonzero(needWorkFeat[5] >= 800)]


    # GPA [mean, sd]
    allGPA_1 = needWorkFeat[6][:]


    # number of Essays [<3,3,4,5,>5] 
    countEssays_1 = [np.count_nonzero(needWorkFeat[7] < 3),np.count_nonzero(needWorkFeat[7] == 3),
        np.count_nonzero(needWorkFeat[7] == 4),np.count_nonzero(needWorkFeat[7] == 5),
        np.count_nonzero(needWorkFeat[7] > 5)]


    # Abroad Credit [<11, 11-20, 21-30, 31-40,>40] 
    countAbroad_1 = [np.count_nonzero(needWorkFeat[8] < 11),
        np.count_nonzero((needWorkFeat[8] >= 11) & (needWorkFeat[8] < 21)),
        np.count_nonzero((needWorkFeat[8] >= 21) & (needWorkFeat[8] < 31)),
        np.count_nonzero((needWorkFeat[8] >= 31) & (needWorkFeat[8] < 41)),
        np.count_nonzero(needWorkFeat[8] >= 41)]


    # AP credits [0-5,6-11,12-18,19-24,25-30,31-36,36>] 
    countAPcred_1 = [np.count_nonzero(needWorkFeat[9] < 6), 
        np.count_nonzero((needWorkFeat[9] >= 6) & (needWorkFeat[9] < 12)),
        np.count_nonzero((needWorkFeat[9] >= 12) & (needWorkFeat[9] < 19)),
        np.count_nonzero((needWorkFeat[9] >= 19) & (needWorkFeat[9] < 25)),
        np.count_nonzero((needWorkFeat[9] >= 25) & (needWorkFeat[9] < 31)),
        np.count_nonzero((needWorkFeat[9] >= 31) & (needWorkFeat[9] < 37)),
        np.count_nonzero(needWorkFeat[9] >= 37)]


    # CS credit [0-5,6-12,13-20,21-40,>40] 
    countCScred_1 = [np.count_nonzero(needWorkFeat[10] < 6),
        np.count_nonzero((needWorkFeat[10] >= 6) & (needWorkFeat[10] < 13)),
        np.count_nonzero((needWorkFeat[10] >= 13) & (needWorkFeat[10] < 21)),
        np.count_nonzero((needWorkFeat[10] >= 21) & (needWorkFeat[10] < 41)),
        np.count_nonzero(needWorkFeat[10] >= 41)]


    # English credit [0-5,6-12,13-20,21-40,>40]
    countEngcred_1 = [np.count_nonzero(needWorkFeat[11] < 6),
        np.count_nonzero((needWorkFeat[11] >= 6) & (needWorkFeat[11] < 13)),
        np.count_nonzero((needWorkFeat[11] >= 13) & (needWorkFeat[11] < 21)),
        np.count_nonzero((needWorkFeat[11] >= 21) & (needWorkFeat[11] < 41)),
        np.count_nonzero(needWorkFeat[11] >= 41)]

    # Science credit [0-5,6-11,12-20,21-30,31-40,41-60,>60]
    countScicred_1 = [np.count_nonzero(needWorkFeat[12] < 6),
        np.count_nonzero((needWorkFeat[12] >= 6) & (needWorkFeat[12] < 12)),
        np.count_nonzero((needWorkFeat[12] >= 12) & (needWorkFeat[12] < 21)),
        np.count_nonzero((needWorkFeat[12] >= 21) & (needWorkFeat[12] < 31)),
        np.count_nonzero((needWorkFeat[12] >= 31) & (needWorkFeat[12] < 41)),
        np.count_nonzero((needWorkFeat[12] >= 41) & (needWorkFeat[12] < 61)),
        np.count_nonzero(needWorkFeat[12] >= 61)]


    # Writing credit [mean,sd]
        #Gaussian Model
    allWrite_1 = needWorkFeat[13][:]
        # Categorical
    countWrite_1 = [np.count_nonzero(needWorkFeat[13] < 6),
        np.count_nonzero((needWorkFeat[13] >= 6) & (needWorkFeat[13] < 13)),
        np.count_nonzero((needWorkFeat[13] >= 13) & (needWorkFeat[13] < 21)),
        np.count_nonzero((needWorkFeat[13] >= 21) & (needWorkFeat[13] < 41)),
        np.count_nonzero(needWorkFeat[13] >= 41)]




    ### Given No Need Work (feature_0) ###

    # Minnesota [0,1]
    countMn_0 = [np.count_nonzero(noWorkFeat[1] == 0), np.count_nonzero(noWorkFeat[1] == 1)]

    # International [0,1]
    countInt_0 = [np.count_nonzero(noWorkFeat[2] == 0), np.count_nonzero(noWorkFeat[2] == 1)]

    # Birth Year [#<1982, #=1982, #=1983, #=1984, #=1985, #>1985]
    countYear_0 = [np.count_nonzero(noWorkFeat[3] < 1982),np.count_nonzero(noWorkFeat[3] == 1982),
    np.count_nonzero(noWorkFeat[3] == 1983),np.count_nonzero(noWorkFeat[3] == 1984),
    np.count_nonzero(noWorkFeat[3] == 1985),np.count_nonzero(noWorkFeat[3] > 1985)]

    # Verbal SAT [mean, sd]
        #Gaussian Model
    allVerbal_0 = noWorkFeat[4][:]
        #Categorical 
    countVerbal_0 = [np.count_nonzero(noWorkFeat[4] < 300), 
                        np.count_nonzero((noWorkFeat[4] >= 300) & (noWorkFeat[4] < 400)),
                        np.count_nonzero((noWorkFeat[4] >= 400) & (noWorkFeat[4] < 500)), 
                        np.count_nonzero((noWorkFeat[4] >= 500) & (noWorkFeat[4] < 600)),
                        np.count_nonzero((noWorkFeat[4] >= 600) & (noWorkFeat[4] < 700)),
                        np.count_nonzero((noWorkFeat[4] >= 700) & (noWorkFeat[4] < 800)),
                        np.count_nonzero(noWorkFeat[4] >= 800)]
    
    # Math SAT [mean, sd]
        #Gaussian Model
    allMath_0 = noWorkFeat[5][:]
        #Categorical
    countMath_0 = [np.count_nonzero(noWorkFeat[5] < 300), 
                        np.count_nonzero((noWorkFeat[5] >= 300) & (noWorkFeat[5] < 400)),
                        np.count_nonzero((noWorkFeat[5] >= 400) & (noWorkFeat[5] < 500)), 
                        np.count_nonzero((noWorkFeat[5] >= 500) & (noWorkFeat[5] < 600)),
                        np.count_nonzero((noWorkFeat[5] >= 600) & (noWorkFeat[5] < 700)),
                        np.count_nonzero((noWorkFeat[5] >= 700) & (noWorkFeat[5] < 800)),
                        np.count_nonzero(noWorkFeat[5] >= 800)]

    # GPA [mean, sd]
    allGPA_0 = noWorkFeat[6][:]

    # number of Essays = [<3,3,4,5,>5] 
    countEssays_0 = [np.count_nonzero(noWorkFeat[7] < 3),np.count_nonzero(noWorkFeat[7] == 3),
        np.count_nonzero(noWorkFeat[7] == 4),np.count_nonzero(noWorkFeat[7] == 5),
        np.count_nonzero(noWorkFeat[7] > 5)]

    # Abroad Credit [<11, 11-20, 21-30, 31-40,>40] 
    countAbroad_0 = [np.count_nonzero(noWorkFeat[8] < 11),
        np.count_nonzero((noWorkFeat[8] >= 11) & (noWorkFeat[8] < 21)),
        np.count_nonzero((noWorkFeat[8] >= 21) & (noWorkFeat[8] < 31)),
        np.count_nonzero((noWorkFeat[8] >= 31) & (noWorkFeat[8] < 41)),
        np.count_nonzero(noWorkFeat[8] >= 41)]


    # AP credits [0-5,6-11,12-18,19-24,25-30,31-36,36>] 
    countAPcred_0 = [np.count_nonzero(noWorkFeat[9] < 6), 
        np.count_nonzero((noWorkFeat[9] >= 6) & (noWorkFeat[9] < 12)),
        np.count_nonzero((noWorkFeat[9] >= 12) & (noWorkFeat[9] < 19)),
        np.count_nonzero((noWorkFeat[9] >= 19) & (noWorkFeat[9] < 25)),
        np.count_nonzero((noWorkFeat[9] >= 25) & (noWorkFeat[9] < 31)),
        np.count_nonzero((noWorkFeat[9] >= 31) & (noWorkFeat[9] < 37)),
        np.count_nonzero(noWorkFeat[9] > 37)]

    # CS credit [0-5,6-12,13-20,21-40,>40]
    countCScred_0 = [np.count_nonzero(noWorkFeat[10] < 6),
        np.count_nonzero((noWorkFeat[10] >= 6) & (noWorkFeat[10] < 13)),
        np.count_nonzero((noWorkFeat[10] >= 13) & (noWorkFeat[10] < 21)),
        np.count_nonzero((noWorkFeat[10] >= 21) & (noWorkFeat[10] < 41)),
        np.count_nonzero(noWorkFeat[10] >= 41)]

    # Eng credit [0-5,6-12,13-20,21-40,>40] 
    countEngcred_0 = [np.count_nonzero(noWorkFeat[11] < 6),
        np.count_nonzero((noWorkFeat[11] >= 6) & (noWorkFeat[11] < 13)),
        np.count_nonzero((noWorkFeat[11] >= 13) & (noWorkFeat[11] < 21)),
        np.count_nonzero((noWorkFeat[11] >= 21) & (noWorkFeat[11] < 41)),
        np.count_nonzero(noWorkFeat[11] >= 41)]


    # Science credit [0-5,6-11,12-20,21-30,31-40,41-60,>60] 
    countScicred_0 = [np.count_nonzero(noWorkFeat[12] < 6),
        np.count_nonzero((noWorkFeat[12] >= 6) & (noWorkFeat[12] < 12)),
        np.count_nonzero((noWorkFeat[12] >= 12) & (noWorkFeat[12] < 21)),
        np.count_nonzero((noWorkFeat[12] >= 21) & (noWorkFeat[12] < 31)),
        np.count_nonzero((noWorkFeat[12] >= 31) & (noWorkFeat[12] < 41)),
        np.count_nonzero((noWorkFeat[12] >= 41) & (noWorkFeat[12] < 61)),
        np.count_nonzero(noWorkFeat[12] >= 61)]

    # Writing credit [mean, sd]
        #Gaussian Model
    allWrite_0 = noWorkFeat[13][:]
        #Categorical
    countWrite_0 = [np.count_nonzero(noWorkFeat[13] < 6),
        np.count_nonzero((noWorkFeat[13] >= 6) & (noWorkFeat[13] < 13)),
        np.count_nonzero((noWorkFeat[13] >= 13) & (noWorkFeat[13] < 21)),
        np.count_nonzero((noWorkFeat[13] >= 21) & (noWorkFeat[13] < 41)),
        np.count_nonzero(noWorkFeat[13] >= 41)]

    # Using discrete categories to model SAT scores and writing credit:
    preprocessed = [[countMn_0,countMn_1,cat_mn], [countInt_0,countInt_1,cat_int], [countYear_0, countYear_1, cat_year],
            [countVerbal_0,countVerbal_1,cat_sat], [countMath_0,countMath_1,cat_sat],[allGPA_0,allGPA_1,0.005],
            [countEssays_0,countEssays_1, cat_essay], [countAbroad_0,countAbroad_1, cat_abroad],
            [countAPcred_0,countAPcred_1, cat_ap],[countCScred_0,countCScred_1, cat_cs], 
            [countEngcred_0, countEngcred_1, cat_eng], [countScicred_0,countScicred_1, cat_sci],
            [countWrite_0,countWrite_1,cat_eng], data, countNoWork, countNeedWork]
    return preprocessed   #indexed by [column - 1], where column is the column in the excel sheet in which the features locates



'''
Function to return the count of data in a feature bin in the training data,
given the feature value of the sample and whether it was actually in the class need work or no need work.

Used by function classifyDiscrete()

param: array featData - preprocessed feature data : [[count of each group given NO need work],[count given NEED work],[category definition]]
       int test - the value of the feature to examine
       int whetherNeedWork - 1 for need work, -1 for no need work
return: int groupCount_noNeed - number of portfolios in the same bin given No Need Work
        int groupCount_need - number of portfolios in the same bin given Need Work
'''
def discreteCount(test, featData, whetherNeedWork): 
    # Given the feature and the value, first determing which bin it belongs to
    x = test
    found = False
    for group in range(len(featData[2])):

        if eval(featData[2][group]) == True: #if the value belongs to the bin, return
            groupId = group    #obtain the bin it belongs to
            found = True
            break
    
    if found == False:
        return [0,0]
    # Second, find the count for the desired bin for each class
    groupCount_noNeed = featData[0][groupId]
    groupCount_need = featData[1][groupId]

    # If we are using a training data as test data, adjust the counts
    if (whetherNeedWork == -1):
        groupCount_noNeed -= 1
    else:
        groupCount_need -= 1

    return [groupCount_noNeed, groupCount_need]



'''
Function to compute various conditional probabilities for a given data
Param: array portfolio - test data to classify
       matrix preprocessData - All preprocessed counts of various features
Return: int SumProbNoWork - sum of discrete log probabilities and P(noNeedWork) or P(NeedWork)
'''
def classifyDiscrete(portfolio,preprocessData):
    count_noWork = preprocessData[-2]
    count_needWork = preprocessData[-1]

    #Since the entry is part of the training data, we want to remove it
    if portfolio[-1] == -1:     # if the entry does NOT need work, adjust the class count
        count_noWork -= 1
    else:                       # if the entry NEED work, adjust
        count_needWork -= 1
    
    # P(NoNeedWork)
    probNoWork = np.log(float(count_noWork) / (count_noWork + count_needWork)) #Calculate probability of NOT needing work
    # P(NeedWork)
    probNeedWork = np.log(float(count_needWork) / (count_noWork + count_needWork)) #Calculate probability of NEEDing work
    
    #compute all discrete probabilities
    discreteProb = []   #a list of probability for each feature. [[P(feature1|noNeed),P(feature1|NeedWork)],.(feature2)..,...]
    #Used labels Minnesota, international, Verbal SAT scores, num_essays, Abroad Credits, AP Credits, CS Credits, English Credits, writing credits
    for i in [1,2,4,7,8,9,10,11,13]:   #for each discrete feature, compute conditional probability
        prob = discreteCount(portfolio[i],preprocessData[i-1],portfolio[-1])
        prob[0] = np.log((float(prob[0])+1)/(count_noWork + len(preprocessData[i-1][2])))         #prob given no need work, using add one smoothing
                                                                                                 #preprocessData[i-1][2] is a lis of all categories in the feature
        prob[1] = np.log((float(prob[1])+1)/(count_needWork + len(preprocessData[i-1][2])))       #prob given need work, using add one smoothing

        discreteProb.append(prob)
    
    #Sum up the discrete log probabilities
    SumProbNoWork = 0
    SumProbNeedWork = 0
    for p in discreteProb:
        SumProbNoWork += p[0]
        SumProbNeedWork += p[1]

    #Add the log probability of the respective classes P(noNeedWork) or P(NeedWork)
    SumProbNoWork += probNoWork
    SumProbNeedWork += probNeedWork

    return [SumProbNoWork,SumProbNeedWork]


'''
Function to compute the mean and standard deviation for a list of values
Param: array data - all data points of the continuous feature
Return: parameters for given data
'''
def normalParam(data):
    return [np.mean(data), np.std(data,ddof=1)]

'''
Function to return the probability of being classified as need work or no need work given the value of 
a feature that is treated as normally distributed. P(feature|label)
Param: int test - value of the feature data
       array featData - array of preprocessed feature data
       int whetherNeedWork - 1 for need work, -1 for no need work
Return int normProbNoWork - log probability of no need work
       int normProbNeedWork - log probability of need work
'''
def normalDist(test, featData, whetherNeedWork):
    x = test
    noWorkData = featData[0][:]
    needWorkData = featData[1][:]

    #adjust data and remove the test data
    if (whetherNeedWork == -1):   #if data is originally classified do not need work
        noWorkData = np.delete(noWorkData,np.argwhere(noWorkData == x)[0])
    else:                         #if data is originally classified need work
        needWorkData = np.delete(needWorkData,np.argwhere(needWorkData == x)[0])

    #calculate the normal dis parameter for both classes after adjusting
    noWorkParam = normalParam(noWorkData)
    needWorkParam = normalParam(needWorkData)

    #compute probability by using the difference between cumulative density function (cdf) with corrected boundaries
        #(featData[2] is the value used to correct the boundaries so that the data is continuous at every point)
    normProbNoWork = scipy.stats.norm(noWorkParam[0],noWorkParam[1]).cdf(x + featData[2]) - scipy.stats.norm(noWorkParam[0],noWorkParam[1]).cdf(x - featData[2])
    normProbNeedWork = scipy.stats.norm(needWorkParam[0], needWorkParam[1]).cdf(x + featData[2]) - scipy.stats.norm(needWorkParam[0], needWorkParam[1]).cdf(x - featData[2])
    
    #compute log probabilities
    normProbNoWork = np.log(normProbNoWork)
    normProbNeedWork = np.log(normProbNeedWork)

    return [normProbNoWork, normProbNeedWork]

'''
Function to compute conditional probabilities for a given data
Param: array portfolio - test data to classify
       matrix preprocessData - All preprocessed counts of various categories
Return: int SumProbNoWork - sum of continuous log probabilities
'''
def classifyNormal(portfolio,preprocessData):
    contProb = []
    for ind in [6]: #compute the continuous category (cumGPA)
        prob = normalDist(portfolio[ind], preprocessData[ind-1],portfolio[-1]) #compute probability of being classified as each class
        contProb.append(prob)
    SumProbNoWork = 0
    SumProbNeedWork = 0
    for p in contProb:
        SumProbNoWork += p[0]
        SumProbNeedWork += p[1]

    return [SumProbNoWork,SumProbNeedWork]        


'''
Function to loop through each data in the training data, remove it from the training data and use that data as a test data.
Calculate the accuracy of the classification.
Param: matrix preprocessData - All preprocessed counts of various categories
'''
def leaveOneOut(preprocessData):
    correct = 0
    wrong = [] #list of protfolio labels that are classified incorrectly
    for portfolio in preprocessData[-3]:
        classification = 0
        discreteProb = classifyDiscrete(portfolio, preprocessData)
        normalProb = classifyNormal(portfolio,preprocessData)
        probNoWork = discreteProb[0] + normalProb[0]
        probNeedWork = discreteProb[1] + normalProb[1]

        if (probNoWork > probNeedWork):
            classification = -1
        else:
            classification = 1
        
        if classification == portfolio[-1]:
            correct += 1
        else:
            wrong.append(portfolio[0]) 
    print "The accuracy of classification is: " + str(float(correct)/len(preprocessData[-3])) #print the accuracy of the classifier
    #print "portfolio id not correctly classified: " + str(wrong) #print the portfolio id of the wrongly classified portfolios


def main():
    preprocessData = preprocess()
    leaveOneOut(preprocessData)

main()








