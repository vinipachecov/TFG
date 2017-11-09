#enconding=utf-8

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


cancer_data = pd.read_csv('data.csv')


def M_B_0_1(label):
    if label == 'M':
        return 1
    else:
        return 0
    
    
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



radius_mean =tf.feature_column.numeric_column('radius_mean')
texture_mean =tf.feature_column.numeric_column('texture_mean')
perimeter_mean =tf.feature_column.numeric_column('perimeter_mean')
area_mean =tf.feature_column.numeric_column('area_mean')
smoothness_mean =tf.feature_column.numeric_column('smoothness_mean')
compactness_mean =tf.feature_column.numeric_column('compactness_mean')
concavity_mean =tf.feature_column.numeric_column('concavity_mean')
concave_points_mean =tf.feature_column.numeric_column('concave_points_mean')
symmetry_mean =tf.feature_column.numeric_column('symmetry_mean')
fractal_dimension_mean =tf.feature_column.numeric_column('fractal_dimension_mean')
radius_se =tf.feature_column.numeric_column('radius_se')
texture_se =tf.feature_column.numeric_column('texture_se')
perimeter_se =tf.feature_column.numeric_column('perimeter_se')
area_se =tf.feature_column.numeric_column('area_se')
smoothness_se =tf.feature_column.numeric_column('smoothness_se')
compactness_se =tf.feature_column.numeric_column('compactness_se')
concavity_se =tf.feature_column.numeric_column('concavity_se')
concave_points_se =tf.feature_column.numeric_column('concave_points_se')
symmetry_se =tf.feature_column.numeric_column('symmetry_se')
fractal_dimension_se =tf.feature_column.numeric_column('fractal_dimension_se')
radius_worst =tf.feature_column.numeric_column('radius_worst')
texture_worst =tf.feature_column.numeric_column('texture_worst')
perimeter_worst =tf.feature_column.numeric_column('perimeter_worst')
area_worst =tf.feature_column.numeric_column('area_worst')
smoothness_worst =tf.feature_column.numeric_column('smoothness_worst')
compactness_worst =tf.feature_column.numeric_column('compactness_worst')
concavity_worst =tf.feature_column.numeric_column('concavity_worst')
concave_points_worst =tf.feature_column.numeric_column('concave_points_worst')
symmetry_worst =tf.feature_column.numeric_column('symmetry_worst')
fractal_dimension_worst =tf.feature_column.numeric_column('fractal_dimension_worst')


feat_cols = [radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,
            concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,
            smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,
            radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,
            concave_points_worst,symmetry_worst,fractal_dimension_worst]


labels_to_drop = ['id','diagnosis']
x_data = cancer_data.drop(labels=labels_to_drop,axis=1)
y_data = cancer_data['diagnosis']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=101)


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=10,num_epochs=1000,shuffle=True)


model = tf.estimator.DNNClassifier(hidden_units=[200,200,200],feature_columns=feat_cols,n_classes=2)

model.train(input_fn=input_func,steps=1000)

eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,
                                                      y=y_test,
                                                      batch_size=10,
                                                      num_epochs=1,
                                                      shuffle=False)

results = model.evaluate(eval_input_func)

print(results)


#predições

pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,
                                                      batch_size=10,
                                                      num_epochs=1,
                                                     shuffle=False)

predictions = model.predict(pred_input_func)

my_predictions = list(predictions)

valores_finais_predicao = [pred['class_ids'][0] for pred in my_predictions]

from sklearn.metrics import classification_report, accuracy_score

print(classification_report(y_test,y_pred=valores_finais_predicao))

print('Final Accuracy',accuracy_score(y_test,y_pred=valores_finais_predicao))


# alterando valores mal classificados
# beningnos que foram classificados como maligos -> 2
# malignos que foram classificados como beningnos -> 3

def muda_cores_mal_classificados(df_y_test,df_y_pred):  
    y_pred_cores = valores_finais_predicao
    for i, pred in enumerate(df_y_pred['diagnosis']):

        if(df_y_test['diagnosis'].values[i] == 0 and pred == 1):
            df_y_pred['diagnosis'].values[i] = 2

        if(df_y_test['diagnosis'].values[i] == 1 and pred == 0):
            df_y_pred['diagnosis'].values[i] = 3    
            
            
# Legendas

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


random_vals = np.random.rand(len(y_test))
spaco = np.linspace(2,4,171)

df_y_test = pd.DataFrame(data=y_test)
df_y_pred = pd.DataFrame(data=valores_finais_predicao, columns=['diagnosis'])

muda_cores_mal_classificados(df_y_test,df_y_pred)

x1 = X_test['radius_mean']  + random_vals
y1 = y_test + spaco

x2 = X_test['radius_mean']  + random_vals
y2 = valores_finais_predicao + spaco


# alterando valores mal classificados
# beningnos que foram classificados como maligos -> 2
# malignos que foram classificados como beningnos -> 3
y_pred_cores = valores_finais_predicao
for i, pred in enumerate(y_pred_cores):
    
    if(df_y_test['diagnosis'].values[i] == 0 and pred == 1):
        print('Benigno mal classificado: ',i)
        
    if(df_y_test['diagnosis'].values[i] == 1 and pred == 0):
        print('Maligno mal classificado: ',i)
        
        
        
#Plot dos resultados
        


fig, ax = plt.subplots()


#Impressão dos valores reais
ax.scatter(x1, y1, color=df_y_test['diagnosis'].apply(lambda x: cores_val_reais[x]),  marker='o')


#Impressão dos valores resultantes da classificação
for i, val in enumerate(df_y_pred['diagnosis']):
    if(val == 0 or val == 1):
        ax.scatter(x2.values[i], y2[i], color=cores_val_classificados[df_y_pred['diagnosis'].values[i]],  marker='o')        
    else:        
        ax.scatter(x2.values[i], y2[i], color=cores_val_classificados[df_y_pred['diagnosis'].values[i]], marker='*')



plt.legend(handles=[pontos_verdes,pontos_pretos,pontos_azuis,pontos_vermelhos,pontos_rosas,pontos_dourados])

