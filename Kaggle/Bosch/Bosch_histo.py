# Bosch Kaggle challenge - explo

# import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ABS = 'G:/Olivier/Kaggle/Bosch/'

FTESTCAT = ABS+'test_categorical.csv'
FTESTNUM = ABS+'test_numeric.csv'

FTRAINNUM = ABS+'train_numeric.csv'

FTRAINDATE = ABS+'train_date.csv'

# On ne garde que les 2 000 premiers cas
data = pd.read_csv(FTRAINNUM, nrows=2000)

plt.hist(data.isnull().sum(axis=1),bins=40)
plt.show()

# extraire les 'cas positifs'
dpos = data[ data['Response']==1]
