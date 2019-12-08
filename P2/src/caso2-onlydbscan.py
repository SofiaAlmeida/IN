# -*- coding: utf-8 -*-
"""
Autor:
    Sofía Almeida Bruno
Basada en el código proporcionado por:
    Jorge Casillas
Fecha:
    Noviembre/2019
Contenido:
    Caso de estudio 2"""
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import preprocessing

from sklearn.impute import SimpleImputer
from math import floor

import seaborn as sns

def norm_to_zero_one(df):
    return (df - df.min()) * 1.0 / (df.max() - df.min())

# Devuelve la tabla latex con la información sobre los algoritmos, recibe una lista con las filas de dicha tabla
def tabla_alg(l):
    # Cabecera de la tabla
    print("\\begin{table}[H]")
    print("\centering")
    print("\caption{Resultados caso de estudio 2}")
    print("\label{tab:algorithms2}")
    print("\\begin{tabular}{lrrrr}")
    print("\\toprule")
    print("Algoritmo & Tiempo ($s$) & Calinski-Harabasz & Silhouette & Número de clusters\\\\")
    print("\midrule")

    # Contenido de la tabla
    for s in l:
        print(s)
    
    # Cierre de la tabla
    print("\\bottomrule")
    print("\end{tabular}")
    print("\end{table}")

#'''Tamaño de clusters
def tam_clusters(clusters, show=False):   
    num_clus = []
    tams = []
    # Cuántos objetos caen en cada cluster
    print("Tamaño de cada cluster:")
    size=clusters['cluster'].value_counts()
    for num,i in size.iteritems():
        print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
        num_clus.append(num)
        tams.append(i)
        
    sns.set()
    plt.rcParams.update({'figure.autolayout': True})
    fig, ax = plt.subplots()
    ax.bar(num_clus, tams, align='center', alpha=0.5)
    ax.set(xlabel='Número de cluster', ylabel='Número de elementos')
    fig.savefig("./fig/caso2/DBSCAN2_tam_clusters.png", transparent=False, dpi=80, bbox_inches="tight")
    if show:
        plt.show()

def bar_graph(data, etiq, show=False):   
    nums = []
    tams = []
    # Cuántos objetos caen en cada cluster
    print("Tamaño de " + etiq)
    size=data[etiq].value_counts()
    for num,i in size.iteritems():
        print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(data)))
        nums.append(num)
        tams.append(i)
        
    sns.set()
    plt.rcParams.update({'figure.autolayout': True})
    fig, ax = plt.subplots()
    ax.bar(nums, tams, align='center', alpha=0.5)
    ax.set(xlabel='Valor de ' + etiq, ylabel='Número de elementos')
    fig.savefig("./fig/caso2/"+ etiq + '2_tam.png', transparent=False, dpi=80, bbox_inches="tight")
    if show:
        plt.show()


#'''Scatter matrix
def scatter_matrix(X_alg, show=False):
    print("---------- Preparando el scatter matrix DBSCAN...")

    #se añade la asignación de clusters como columna a X
    sns.set()
    variables = list(X_alg)
    variables.remove('cluster')
    sns_plot = sns.pairplot(X_alg, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") 
    plt.subplots_adjust(wspace=.03, hspace=.03)
    plt.savefig("./fig/caso2/DBSCAN2_scattermatrix.png")
    if show:
        plt.show()
    plt.clf()
    print("")
    #'''

# Distribución
def distplot(X, name, k, usadas, show=False):
    print("\nGenerando distplot..." + name)
    n_var = len(usadas)

    sns.set()
    fig, axes = plt.subplots(k, n_var, sharey=True, figsize=(15,15))
    fig.subplots_adjust(wspace=0.007, hspace = 0.04)

    colors = sns.color_palette(palette=None, n_colors=k, desat=None)

    rango = [] # contendrá el mín y el máx para cada variable de todos los clusters
    for j in range(n_var):
        rango.append([X_alg[usadas[j]].min(), X_alg[usadas[j]].max()])
    
    for i in range(k):
        dat_filt = X_alg.loc[X_alg['cluster']==i]
        for j in range(n_var):
            ax = sns.distplot(dat_filt[usadas[j]], 
color = colors[i], label = "", ax = axes[i,j])
            if (i==k-1):
                axes[i,j].set_xlabel(usadas[j])
            else:
                axes[i,j].set_xlabel("")
        
            if (j==0):
                axes[i,j].set_ylabel("Cluster "+str(i))
            else:
                axes[i,j].set_ylabel("")
       
            axes[i,j].set_yticks([])
            axes[i,j].grid(axis='x', linestyle='-', linewidth='0.2', color='gray')
            axes[i,j].grid(axis='y', b=False)
       
            ax.set_xlim(rango[j][0]-0.05*(rango[j][1]-rango[j][0]), rango[j][1]+0.05*(rango[j][1]-rango[j][0]))
            
    plt.savefig("./fig/caso2/" + name + "2_distplot.png")
    if show:
        plt.show()
    
    plt.clf()
#'''

#''' Box plot
# k = nº cluster
def boxplot(X, name, k, usadas, show=False):
    print("----------Creando boxplot " + name + "-----")
    n_var = len(usadas)

    sns.set()
    fig, axes = plt.subplots(k, n_var, sharey=True, figsize=(17, 17))
    fig.subplots_adjust(wspace=0.04, hspace=0.5)
    
    colors = sns.color_palette(palette=None, n_colors=k, desat=None)

    rango = [] # contendrá el mín y el máx para cada variable de todos los clusters
    for j in range(n_var):
        rango.append([X_alg[usadas[j]].min(), X_alg[usadas[j]].max()])

    for i in range(k):
        dat_filt =  X_alg.loc[X_alg['cluster']==i]
        for j in range(n_var):
            ax = sns.boxplot(dat_filt[usadas[j]], color=colors[i], flierprops={'marker':'o','markersize':4}, ax=axes[i,j])

            if (i==k-1):
                axes[i,j].set_xlabel(usadas[j])
            else:
                axes[i,j].set_xlabel("")
            
            if (j==0):
                axes[i,j].set_ylabel("Cluster "+str(i))
            else:
                axes[i,j].set_ylabel("")
            
            axes[i,j].set_yticks([])
            axes[i,j].grid(axis='x', linestyle='-', linewidth='0.2', color='gray')
            axes[i,j].grid(axis='y', b=False)
            ax.set_xlim(rango[j][0]-0.05*(rango[j][1]-rango[j][0]), rango[j][1]+0.05*(rango[j][1]-rango[j][0]))

    fig.savefig("./fig/caso2/" + name + "2_boxplot.png")
    if show:
        plt.show()
    plt.clf()
#'''

#---------------------------------------------------------
    
# Leemos el csv desde el propio directorio
censo = pd.read_csv('mujeres_fecundidad_INE_2018.csv')

# Imputaremos los valores perdidos con la media
prueba = pd.DataFrame(censo)

for col in censo:
   censo[col].fillna(censo[col].mean(), inplace=True)

#seleccionar casos
subset = censo.loc[(censo['DESEOHIJOS'] == 6)]
print("Subconjunto formado por " + str(len(subset)) + " objetos")
# seleccionar variables de interés para clustering
usadas = ['EDAD', 'SATISFACEVI', 'EDADIDEAL', 'ESTUDIOSA']
X = subset[usadas] # Columnas que voy a estudiar

#Normalizamos
X_normal = X.apply(norm_to_zero_one)

# Ejecuto clustering

# Instanciamos los algoritmos
# Algoritmo DBSCAN
# Parámetros: eps (mínima distancia), min_samples (número de ejemplos en el vecindario de un punto para considerarlo centro)
dbscan = DBSCAN(eps=0.127, min_samples=20)

# Lista de algoritmos a utilizar
algorithms = (('DBSCAN', dbscan)) 
cluster_predict = {}

# Bucle algoritmos
filas_tabla_res = []

print('----- Ejecutando ' + 'DBSCAN', end='') # -----

#Tomamos tiempos
t = time.time()
# Ejecuto el algoritmo y asigno los clusters
cluster_predict['DBSCAN'] = dbscan.fit_predict(X_normal)
tiempo = time.time() - t
#Pinto resultados
print(": {:.2f} segundos, ".format(tiempo), end='')
try:
    metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict['DBSCAN'])
    print("Calinski-Harabasz Index: {:.3f}, ".format(metric_CH), end='')
except:
    print("----ERROR: No podemos calcular el índice Calinski-Harabasz---")
    metric_CH = -1
# Otra medida de rendimiento, menos eficiente
# el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
if len(X) > 10000:
    muestra_silhoutte = 0.2
else:
    muestra_silhoutte = 1.0
    
# Para hacer pruebas podemos hacer un muestreo, para los resultados definitivos ya dejamos que se ejecute   
try:
    metric_SC = metrics.silhouette_score(X_normal, cluster_predict['DBSCAN'], metric='euclidean', sample_size = floor(muestra_silhoutte*len(X)), random_state=123456)
    print("Silhouette Coefficient: {:.5f}".format(metric_SC))
except:
    print("No podemos calcular el índice silhouette")
    metric_SC = -1
        
# Calculamos el número de clusters (será interesante en los algoritmos que no lo imponemos)
labels = dbscan.labels_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("Número de clusters: {:.0f}".format(n_clusters_))


# Fila tabla
s = 'DBSCAN' + " & "
s += "{:.3f}".format(tiempo) + " & "
s += "{:.3f}".format(metric_CH) + " & "
s += "{:.5f}".format(metric_SC) + " & "
s += "{:.0f}".format(n_clusters_) + " \\\\"
    
filas_tabla_res.append(s)

# Visualización
clusters = pd.DataFrame(cluster_predict['DBSCAN'],index=X.index,columns=['cluster'])
X_alg = pd.concat([X, clusters], axis=1)
tam_clusters(clusters)

   
try:
    scatter_matrix(X_alg)
except:
    print("------ERROR: No se pudo crear el scatter matrix------")
try:
    distplot(X_alg, 'DBSCAN', n_clusters_, usadas)
except:
    print("------ERROR: No se pudo crear el distplot-----")
try:
    boxplot(X_alg, 'DBSCAN', n_clusters_, usadas)
except:
    print("------ERROR: No se pudo crear el boxplot------")
#----

tabla_alg(filas_tabla_res)
