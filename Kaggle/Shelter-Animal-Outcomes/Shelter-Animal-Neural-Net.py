import os

import numpy as np
import pandas as pd
import pybrain
import pickle

import matplotlib.pyplot as plt

# Modules for Neural Network
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
# Paths for csv data

#FTRAIN= '~/Dropbox/Travail/Big-data/Kaggle/Shelter-Animal/train.csv'
#FTRAIN = 'C:/Users/Trivy/Documents/Dropbox/Travail/Big-data/Kaggle/Shelter-Animal/train.csv'
FTRAIN = 'train.csv'

#FTEST= '~/Dropbox/Travail/Big-data/Kaggle/Shelter-Animal/test.csv'
#FTEST = 'C:/Users/Trivy/Documents/Dropbox/Travail/Big-data/Kaggle/Shelter-Animal/test.csv'
FTEST = 'test.csv'

#FEXPORT= '~/Dropbox/Travail/Big-data/Kaggle/Shelter-Animal/Export-check.csv'
#FEXPORT = 'C:/Users/Trivy/Documents/Dropbox/Travail/Big-data/Kaggle/Shelter-Animal/Export5.csv'
FEXPORT = 'Export-check.csv'

#FTRAINEDNN = '~/Dropbox/Travail/Big-data/Kaggle/Shelter-Animal/Trained_net2.xml'
FTRAINEDNN = 'C:/Users/Trivy/Documents/Dropbox/Travail/Big-data/Kaggle/Shelter-Animal/Trained_net2.xml'
# FTRAINEDNN = 'Trained_net2.xml'

# Introduce the different possible outcomes
Outcomes=[
    'Adoption',
    'Died',
    'Euthanasia',
    'Return_to_owner',
    'Transfer'
    ]

Nb_Outcomes=len(Outcomes)

# def GenLabels(df):
#     df['Label']=\
#     df['Adoption']\
#     +2*df['Died']\
#     +3*df['Euthanasia']\
#     +4*df['Return_to_owner']
#     +5*df['Transfer']
#     return df
# # Careful! the order of the labels matters!

# Introduce names for the features
listFeat=[
    'feat1',
    'feat2',
    'feat3',
    'feat4',
    ]

Nb_listFeat=len(listFeat)

def GenFeatures(df):
    for i in range(Nb_listFeat):
        df[listFeat[i]]=np.log(df['AgeNb'])**((i+1)/2.)
    return df
# Careful! the order of the labels matters!

def preprocess(df, OptArg=False):
    # The optional argument 'OptArg' indicates whether we should evaluate the outcome columns (train case)
    
    
    # ==============================================
    # Preprocessing: 'Named', 'IsCat', 'AgeNb'
    # ==============================================
    
    # Create and evaluate the column 'HasName'
    df['Named']=(df['Name'].notnull())
    
    # Create and evaluate the column 'IsCat'
    df['IsCat']=(df['AnimalType']=='Cat')
    
    # Express all 'AgeuponOutcome' in days in 'AgeNb'
    #
    # From Andy's script (https://www.kaggle.com/andraszsom/shelter-animal-outcomes/age-gender-breed-and-name-vs-outcome),
    # slightly modified: if the entry is "Nan", returns Nan.
    def age_to_days(item):
        # convert item to list if it is one string
        if type(item) is str:
            item = [item]
        ages_in_days = np.zeros(len(item))
        for i in range(len(item)):
            # check if item[i] is str
            if type(item[i]) is str:
                if 'day' in item[i]:
                    ages_in_days[i] = int(item[i].split(' ')[0])
                if 'week' in item[i]:
                    ages_in_days[i] = int(item[i].split(' ')[0])*7
                if 'month' in item[i]:
                    ages_in_days[i] = int(item[i].split(' ')[0])*30
                if 'year' in item[i]:
                    ages_in_days[i] = int(item[i].split(' ')[0])*365    
            else:
                # item[i] is not a string but a nan
                ages_in_days[i] = np.nan
        return ages_in_days
    
    df['AgeNb']=np.maximum(2,age_to_days(df['AgeuponOutcome']))
    # Those with 0 age are send to 2 days (there are many animals with 'AgeNb' <=10)
        
    # sum(df['AgeNb'].isnull()) yields 18
    # Replace by the average age:
    mean_age=df['AgeNb'].mean()
    
    df['AgeNb'].fillna(mean_age,inplace=True)
    
    for i in range(Nb_listFeat):
        df[listFeat[i]]=np.log(df['AgeNb'])**(i/2)
    
    # Create the different Outcome columns
    for i in range(len(Outcomes)):
        df[Outcomes[i]]=0
    
    # ==================================================
    # Preprocessing 'Sex'
    # ==================================================
    # Extract "male/female" and 'Intact/not" from gender
    
    # Fill in missing 'SexuponOutcome' with 'Unknown'
    df['SexuponOutcome'].fillna('Unknown',inplace=True)
    
    # Create column 'IsIntact'
    df['IsIntact']=(df['SexuponOutcome'].str.contains('Intact'))
    
    # Create column 'IsFemale'
    df['IsFemale']=(df['SexuponOutcome'].str.contains('Female'))
    
    # NB: implicitly, if df['SexuponOutcome']=='Unknown', then the case is treated as "Neutered Male" (as this is the most common designation among the 4 possible cases).
    
    if OptArg :
        # Evaluate the different Outcome columns
        for i in range(len(Outcomes)):
            dftrain[Outcomes[i]]=(dftrain['OutcomeType']==Outcomes[i])
        df=GenFeatures(df)
    return df

# ==================================================
# Execution (part 1)
# ==================================================

# Training section:
dftrain = pd.read_csv(FTRAIN,header=0)
# reads the train file and integrate it into a Pandas dataframe.
# sep=';' to use only ";" as delimiters
# decimal=',' to use "," as decimal point (and not "." (default behavior))

dftrain=preprocess(dftrain,True)

# ==================================================
# Neural network model: using previous preprocessing
# ==================================================

listFeatN=[
    'feat1N',
    'feat2N',
    'feat3N',
    'feat4N',
    ]

def normalize_def(df):
    X_mean=df.ix[:,listFeat].mean().values
    X_delta=(df.ix[:,listFeat].max(axis=0) - df.ix[:,listFeat].min(axis=0)).values
    for i in range(Nb_listFeat):
        df.ix[:,listFeatN[i]]=2*(df.ix[:,listFeat[i]] - X_mean[i])/X_delta[i]
    return df, X_mean, X_delta
    
def normalize(df,X_mean,X_delta):
    for i in range(Nb_listFeat):
        df.ix[:,listFeatN[i]]=2*(df.ix[:,listFeat[i]] - X_mean[i])/X_delta[i]
    return df

# Introduce the columns we want to use for the Neural network
listNN=[
    'Named',
    'IsCat',
    'IsIntact',
    'IsFemale'
    ]
# 'AgeNb', mais mauvaise normalisation...

listNN=listNN+listFeatN

Nb_features=len(listNN)

def extract_X(df):
    # Returns X (extracted features) 
    X=df.ix[:,listNN].astype(np.float32).values
    return X

def extract_y(df):
    # Returns y (extracted features) 
    y=df.ix[:,Outcomes].values
    #N_y=len(y)
    #np.array(y).astype(np.float32).reshape((N_y,1))
    return np.array(y).astype(np.float32)    

def loadInputTarget(ia,ta):
    ds = SupervisedDataSet(Nb_features, Nb_Outcomes)
    # assert(ia.shape[0] == ta.shape[0])
    ds.setField('input', ia)
    ds.setField('target', ta)
    return ds

# ==================================================
# Execution (part 2)
# ==================================================

# Neural Network properly speaking:
#
net = buildNetwork(Nb_features, 25, 25, Nb_Outcomes, outclass=SoftmaxLayer, bias=True)

# dftrain defined previously
dftrain, X_mean, X_delta=normalize_def(dftrain)
X_train = extract_X(dftrain)
y_train = extract_y(dftrain)

ds = loadInputTarget(X_train,y_train)
# NB: preprocessing done previously

# Splitting of the data, keep only 1/10 of all data points (for quicker computations, in order to experiment)
ds_red, ds_extra = ds.splitWithProportion( 0.1 )

# NB: we can see what is "inside" ds_red by: ds_red.data. To check the "size" of ds_red: ds_red['input'].shape

trainer = BackpropTrainer(net, ds_red,learningrate=0.01,verbose=True)

trainer.trainUntilConvergence(verbose=True,maxEpochs=200)


# trainer.trainEpochs(20)

# trainer.trainUntilConvergence(verbose=True,maxEpochs=1)

# # To save a trained network:

##   fileObject = open(FTRAINEDNN, 'w')
 # pickle.dump(net, fileObject)
 # fileObject.close()
#
# # To recover a trained network:
#
# fileObject = open(FTRAINEDNN,'r')
# net = pickle.load(fileObject)

# =========================================
# Prediction section:
#
# Read test file
dftest = pd.read_csv(FTEST,header=0)

dftest=preprocess(dftest)

# Normalization of features
dftest=normalize(dftest,X_mean,X_delta)
X_test = extract_X(dftest)



y_test=np.zeros((len(X_test),Nb_Outcomes))
ds1 = loadInputTarget(X_test,y_test)

Output=net.activateOnDataset(ds1)

dftest.ix[:,Outcomes]=Output
# NB: the columns of "Outcomes" were already created in "preprocess(dftest)"
# 
# 
# # =================================================
# # # Export of results in expected submission format
# # # =================================================
# #         
listExport=[\
    'ID',
    'Adoption',
    'Died',
    'Euthanasia',
    'Return_to_owner',
    'Transfer'
]



dExport=dftest.ix[:,listExport]

dExport.to_csv(FEXPORT, index=False)
# 
# # =================================================
# # Conversion in Numpy array
# # =================================================
# 
# # # Convert to a Numpy array, getting ride of first column
# # Export=dExport.ix[:,1:6].values
# 