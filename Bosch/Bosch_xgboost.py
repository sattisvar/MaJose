# Bosch Kaggle challenge - a first script

# Script from joconnor (https://www.kaggle.com/joconnor/bosch-production-line-performance/python-xgboost-starter-0-209-public-mcc/notebook)

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.cross_validation import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# Data divided into several files
ABS = 'E:/Kaggle/Bosch/csv/'

FTESTCAT = ABS+'test_categorical.csv'
FTESTNUM = ABS+'test_numeric.csv'
FTESTDATE = ABS+'test_date.csv'

FTRAINCAT = ABS+'train_categorical.csv'
FTRAINNUM = ABS+'train_numeric.csv'
FTRAINDATE = ABS+'train_date.csv'

# I'm limited by RAM here and taking the first N rows is likely to be
# a bad idea for the date data since it is ordered.
# Sample the data in a roundabout way:
date_chunks = pd.read_csv(FTRAINDATE, index_col=0, chunksize=100000, dtype=np.float32)
num_chunks = pd.read_csv(FTRAINNUM, index_col=0,
                         usecols=list(range(969)), chunksize=100000, dtype=np.float32)
X = pd.concat([pd.concat([dchunk, nchunk], axis=1).sample(frac=0.05)
               for dchunk, nchunk in zip(date_chunks, num_chunks)])
y = pd.read_csv(FTRAINNUM, index_col=0, usecols=[0,969], dtype=np.float32).loc[X.index].values.ravel()
X = X.values

clf = XGBClassifier(base_score=0.005)
clf.fit(X, y)

# threshold for a manageable number of features
plt.hist(clf.feature_importances_[clf.feature_importances_>0])
important_indices = np.where(clf.feature_importances_>0.005)[0]
print(important_indices)

# prediction:
iter_num_csv= pd.read_csv(FTESTNUM, index_col=0, usecols=list(range(969)), iterator=True, chunksize=10000,dtype=np.float32)
iter_date_csv= pd.read_csv(FTESTDATE, index_col=0, iterator=True, chunksize=10000, dtype=np.float32)

y_test = np.concatenate([clf.predict(pd.concat([dchunk,nchunk], axis=1).values) for dchunk, nchunk in zip(iter_date_csv, iter_num_csv)])

# Load and export submission sample
FSAMPLESUB = ABS+'sample_submission.csv'
FEXPORT = ABS+'submission_xgboost1.csv'

submission = pd.read_csv(FSAMPLESUB)

submission['Response']=y_test.astype(int)

submission.to_csv(FEXPORT, index=False)