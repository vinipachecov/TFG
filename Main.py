# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import pandas as pd
import numpy as np

# Criando setup

dataframe = pd.read_csv('/home/vinicius/Repos/TFG/data.csv')
X = dataframe.iloc[:, 2:-1].values

# 
Y = dataframe.iloc[:, 1].values
Y = np.matrix(dataframe.iloc[:, 1].values)



# Encoding categorical data
#Maligo = 1  | Benigno = 0
for i in range(Y.shape[1]):
    if Y[0,i] == 'M':
        Y[0,i] = 1
    else:
        Y[0,i] = 0
        
        
    
