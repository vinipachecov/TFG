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

plt.title('Classificacao vs Dados Reais')
#
pontos_verdes = mlines.Line2D([], [], color='green', marker='o',
                          markersize=15, label='Benignos Reais')

pontos_pretos = mlines.Line2D([], [], color='black', marker='o',
                          markersize=15, label='Malignos Reais')

pontos_azuis = mlines.Line2D([], [], color='blue', marker='o',
                          markersize=15, label='Benignos Classificados')

pontos_vermelhos = mlines.Line2D([], [], color='red', marker='o',
                          markersize=15, label='Malignos classificados')

plt.legend(handles=[pontos_verdes,pontos_pretos,pontos_azuis,pontos_vermelhos])

#criar um space para adicionar aos beningnos e malignos e distanciar as amostras para plotagem





# valores reais da amostra
benignos_amostra = []
malignos_amostra = []
print_x_test_f1_benignos_amostra = []
print_x_test_f2_benignos_amostra = []

print_x_test_f1_malignos_amostra = []
print_x_test_f2_malignos_amostra = []

#Impressao dos valores classificados
benignos = []
malignos = []
print_x_test_f1_benignos = []
print_x_test_f2_benignos = []

print_x_test_f1_malignos = []
print_x_test_f2_malignos = []

for i in y_test:
    if(i == 0):
        benignos_amostra.append(i)
        print_x_test_f1_benignos_amostra.append(X_test['radius_mean'].values[i])
        print_x_test_f2_benignos_amostra.append(X_test['radius_worst'].values[i])
    else:
        malignos_amostra.append(i)
        print_x_test_f1_malignos_amostra.append(X_test['radius_mean'].values[i])
        print_x_test_f2_malignos_amostra.append(X_test['radius_worst'].values[i])    

for i in y_pred:
    if(i == 0):
        benignos.append(i)
        print_x_test_f1_benignos.append(X_test['radius_mean'].values[i])
        print_x_test_f2_benignos.append(X_test['radius_worst'].values[i])
    else:
        malignos.append(i)
        print_x_test_f1_malignos.append(X_test['radius_mean'].values[i])
        print_x_test_f2_malignos.append(X_test['radius_worst'].values[i])

        
benignos_add_values = np.random.random([len(benignos)])
spaco_add_values2 = np.random.random([len(benignos)])

malignos_add_values = np.random.random([len(malignos_amostra)]) * 1.5
spaco_add_values_mal = np.random.random([len(malignos_amostra)]) * 1.5

dif_benigos_amostra_classificados = len(benignos)-len(benignos_amostra)        
print (dif_benigos_amostra_classificados)
total_benignos = len(benignos_amostra) 


plt.plot(print_x_test_f1_benignos_amostra + benignos_add_values[:total_benignos],
         benignos_amostra + spaco_add_values2[:total_benignos],'go')

plt.plot(print_x_test_f1_malignos_amostra + malignos_add_values, 
         malignos_amostra + spaco_add_values_mal, 'ko')            

plt.plot(print_x_test_f1_benignos + benignos_add_values[:107],
         benignos + spaco_add_values2[:107],'bo')
plt.plot(print_x_test_f1_malignos + malignos_add_values[:64],
         malignos + spaco_add_values_mal[:64], 'ro')            


index = list(range(len(X_test)))
X_test = X_test.assign(id = index)

import matplotlib.pyplot as plt
import numpy as np

cores_val_reais = {0:'g',1:'k'}
cores_val_classificados={0:'b',1:'r'}


# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api


import matplotlib.patches as mpatches




pontos_verdes = mlines.Line2D([], [], color='green', marker='o',
                          markersize=7, label='Benignos Reais')

pontos_pretos = mlines.Line2D([], [], color='black', marker='o',
                          markersize=7, label='Malignos Reais')

pontos_azuis = mlines.Line2D([], [], color='blue', marker='o',
                          markersize=7, label='Benignos Classificados')

pontos_vermelhos = mlines.Line2D([], [], color='red', marker='o',
                          markersize=7, label='Malignos classificados')



cores_val_reais = {0:'g',1:'k'}
cores_val_classificados={0:'b',1:'r'}


# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api

random_vals = np.random.rand(len(y_test))
spaco = np.linspace(2,4,171)

df_y_test = pd.DataFrame(data=y_test)
df_y_pred = pd.DataFrame(data=y_pred, columns=['diagnosis'])


fig, ax = plt.subplots()

labels = ['Benignos Reais', 'Malignos Reais' ,'Benignos Classificados', 'Malignos classificados' ]

#fig.legend(handles=[pontos_verdes,pontos_pretos,pontos_azuis,pontos_vermelhos],labels=labels)

s = 121


x1 = X_test['radius_mean']  + random_vals
y1 = y_test + spaco

x2 = X_test['radius_mean']  + random_vals
y2 = y_pred + spaco


ax.scatter(x1, y1, color=df_y_test['diagnosis'].apply(lambda x: cores_val_reais[x]),  marker='o')

ax.scatter(x2, y2, color=df_y_pred['diagnosis'].apply(lambda x: cores_val_classificados[x]),  marker='o')



plt.legend(handles=[pontos_verdes,pontos_pretos,pontos_azuis,pontos_vermelhos] , bbox_to_anchor=(1.1, 1))

