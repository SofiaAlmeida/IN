# -*- coding: utf-8 -*-
"""
Autor:
    Sofía Almeida Bruno
Fecha:
    Diciembre/2019
Contenido:
    Visualización de los datos DrivenData:
       https://www.drivendata.org/competitions/57/nepal-earthquake/
    Inteligencia de Negocio
    Grado en Ingeniería Informática
    Universidad de Granada
"""

import pandas as pd
import numpy as np
import time
from sklearn import preprocessing
import matplotlib.pyplot as plt # plotting
import seaborn as sns

# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    sns.set()
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.savefig("../fig/"+ str(nGraphShown) + '_dist.png', transparent=False, dpi=80, bbox_inches="tight")
    plt.show()


#'''Tamaño de clases
def class_size(df): 
    # Cuántos objetos caen en cada clase
    print("Tamaño de cada clase:")
    size=df['damage_grade'].value_counts()
    for num,i in size.iteritems():
        print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(df)))

# Matriz de dispersión
def scatter_matrix(df, df_labels, selected_features, name):
    subset = df[selected_features]

    sns.set()
    print("preimagen")
    sns.pairplot(df.join(df_labels), hue='damage_grade')
    print("postimagen")
    sns.savefig("../fig/"+ name + '_scatter.png')

# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    sns.set()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.savefig("../fig/corr_matrix.svg")
    
#------------------------------------------------------------------
#------------------------------------------------------------------
#------------------------------------------------------------------
'''
lectura de datos
'''
#los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
data_x = pd.read_csv('../data/nepal_earthquake_tra.csv')
data_y = pd.read_csv('../data/nepal_earthquake_labels.csv')
data_x_tst = pd.read_csv('../data/nepal_earthquake_tst.csv')
data_x.dataframeName = 'train_values'


#se quitan las columnas que no se usan
data_x.drop(labels=['building_id'], axis=1,inplace = True)
data_x_tst.drop(labels=['building_id'], axis=1,inplace = True)
data_y.drop(labels=['building_id'], axis=1,inplace = True)

#plotPerColumnDistribution(data_y, len(data_y), 1)
#class_size(data_y)

#data_x.info()
''' is any missing values across columns'''
#print(data_x.isnull().any())
#scatter_matrix(data_x, data_y, ['count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage', 'position'], "1")
#plotCorrelationMatrix(data_x, 12)
print(data_x.describe().T.style.background_gradient(cmap='Set2',low =0.4,high=0.1,axis=0))
