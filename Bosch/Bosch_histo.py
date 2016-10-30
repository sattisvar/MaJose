# Bosch Kaggle challenge - explo

# import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FTESTCAT = 'G:/Olivier/Kaggle/Bosch/test_categorical.csv'
FTESTNUM = 'G:/Olivier/Kaggle/Bosch/test_numeric.csv'

FTRAINNUM = 'G:/Olivier/Kaggle/Bosch/train_numeric.csv'

FTRAINDATE = 'G:/Olivier/Kaggle/Bosch/train_date.csv'

# On ne garde que les 50 000 premiers cas
data = pd.read_csv(FTRAINNUM, nrows=2000)

plt.hist(data.isnull().sum(axis=1),bins=40)
plt.show()

# extraire les 'cas positifs'
dpos = data[ data['Response']==1]
