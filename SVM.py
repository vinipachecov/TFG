#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:31:28 2017

@author: vinicius
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#encoding the y-label
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
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#results
print(classification_report(y_test,y_pred=y_pred))


#printing the values

#creating a space
add_values = np.linspace(0,3,171)

#adding some values because of the binary values
y_values = y_pred + add_values

plt.figure(1)
# print only the normal data
plt.scatter(x=X_test[:,0] + add_values, y=y_test +add_values)

#predicted values
y_predt = y_pred + add_values

plt.plot(X_test[:,0] + add_values, y_test +add_values, 'ro', 
         X_test[:,0] + add_values, y_predt, 'mo')

plt.show()
#importing colormap