# Bosch challenge Kaggle - extraction des "cas positifs"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ABS = 'G:/Olivier/Kaggle/Bosch/'

FTESTCAT = ABS+'test_categorical.csv'
FTESTNUM = ABS+'test_numeric.csv'

FTRAINNUM = ABS+'train_numeric.csv'

FTRAINDATE = ABS+'train_date.csv'

FPOS = ABS+'pos_numeric2.csv'


iter_csv= pd.read_csv(FTRAINNUM, iterator=True, chunksize=10000)

dpos = pd.concat([chunk[chunk['Response']==1] for chunk in iter_csv])

dpos.to_csv(FPOS)