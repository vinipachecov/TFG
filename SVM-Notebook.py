#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 19:06:00 2017

@author: vinicius
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.lines as mlines


def M_B_0_1(label):
    if label == 'M':
        return 1
    else:
        return 0
    

cancer_data = pd.read_csv('data.csv')
cancer_data['diagnosis'] = cancer_data['diagnosis'].apply(M_B_0_1)



cols_normalizar = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave_points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

cancer_data[cols_normalizar] = cancer_data[cols_normalizar].apply(lambda x: (x - np.mean(x)) / (np.std(x)  ) )


labels_to_drop = ['id','diagnosis']
x_data = cancer_data.drop(labels=labels_to_drop,axis=1)
y_data = cancer_data['diagnosis']


X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=101)



from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)


print(classification_report(y_test,y_pred=y_pred))



# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api



import matplotlib.patches as mpatches


# alterando valores mal classificados
# beningnos que foram classificados como maligos -> 2
# malignos que foram classificados como beningnos -> 3

def muda_cores_mal_classificados(df_y_test,df_y_pred):  
    y_pred_cores = y_pred
    for i, pred in enumerate(df_y_pred['diagnosis']):

        if(df_y_test['diagnosis'].values[i] == 0 and pred == 1):
            df_y_pred['diagnosis'].values[i] = 2

        if(df_y_test['diagnosis'].values[i] == 1 and pred == 0):
            df_y_pred['diagnosis'].values[i] = 3


marcadores = {0:'o',1:'o',2:'*',3:'*'}

pontos_verdes = mlines.Line2D([], [], color='green', marker='o',
                          markersize=7, label='Benignos Reais')

pontos_pretos = mlines.Line2D([], [], color='black', marker='o',
                          markersize=7, label='Malignos Reais')

pontos_azuis = mlines.Line2D([], [], color='blue', marker='o',
                          markersize=7, label='Benignos Classificados')

pontos_vermelhos = mlines.Line2D([], [], color='red', marker='o',
                          markersize=7, label='Malignos classificados')

pontos_rosas = mlines.Line2D([], [], color='fuchsia', marker='*',
                          markersize=7, label='Benignos*')

pontos_dourados = mlines.Line2D([], [], color='saddlebrown', marker='*',
                          markersize=7, label='Malignos*')

# Verde benigno
#Preto maligno
cores_val_reais = {0:'g',1:'k'}

#Azul - benigno corretamente classifica
# Vermelho - maligno corretamente classificado
# Fuchsia - benigno incorretamente classificado (era benigno, mas foi classificado como maligno)
# marrom - maligno incorretamente classificado (era maligno, mas foi classificado como beningno)
cores_val_classificados={0:'b',1:'r',2:'fuchsia',3:'saddlebrown'}

#TRATAMENTO DOS DADOS

random_vals = np.random.rand(len(y_test))
spaco = np.linspace(2,4,171)

df_y_test = pd.DataFrame(data=y_test)
df_y_pred = pd.DataFrame(data=y_pred, columns=['diagnosis'])

muda_cores_mal_classificados(df_y_test,df_y_pred)

x1 = X_test['radius_mean']  + random_vals
y1 = y_test + spaco

x2 = X_test['radius_mean']  + random_vals
y2 = y_pred + spaco




fig, ax = plt.subplots()


#Impressão dos valores reais
ax.scatter(x1, y1, color=df_y_test['diagnosis'].apply(lambda x: cores_val_reais[x]),  marker='o')


#Impressão dos valores resultantes da classificação
for i, val in enumerate(df_y_pred['diagnosis']):
    if(val == 0 or val == 1):
        ax.scatter(x2.values[i], y2[i], color=cores_val_classificados[df_y_pred['diagnosis'].values[i]],  marker='o')        
    else:        
        ax.scatter(x2.values[i], y2[i], color=cores_val_classificados[df_y_pred['diagnosis'].values[i]], marker='*')



plt.legend(handles=[pontos_verdes,pontos_pretos,pontos_azuis,pontos_vermelhos,pontos_rosas,pontos_dourados] , bbox_to_anchor=(1.05, 1))

