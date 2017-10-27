#coding=utf-8

import matplotlib.pyplot as plt
import pandas as pd
#lê o arquivo com os dados
arquivo = pd.read_csv('teste.csv')
#lê as colunas para o plot em 2D
plt.ylabel(arquivo.columns[2])
plt.xlabel(arquivo.columns[0])
#define cor vermelha para diagnóstico B=benigno e azul para M=maligno
colors = {'B':'red','M':'blue'}
#monta os dados conforme o tipo de diagnóstico, usando a definição anterior de colors através do lambda
plt.scatter(arquivo['ID'].values,arquivo['radius_mean'].values, c=arquivo['diagnosis'].apply(lambda x: colors[x]))
#mostra o gráfico 2D
plt.show()

