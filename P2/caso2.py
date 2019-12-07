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

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy

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
    fig.savefig("./fig/caso2/"+ name + '_tam_clusters.png', transparent=False, dpi=80, bbox_inches="tight")
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
    fig.savefig("./fig/caso2/"+ etiq + '_tam.png', transparent=False, dpi=80, bbox_inches="tight")
    if show:
        plt.show()
        
# Heatmap        
def heatmap(centers, show=False):
    print("---------- Preparando el heatmap...")
    centers_desnormal = centers.copy()
    
        # Se convierten los centros a los rangos originales antes de normalizar
    for var in list(centers):
        centers_desnormal[var] = X[var].min() + centers[var] * (X[var].max() - X[var].min())

    sns.set(font_scale=2)
    # Creamos el mapa de temperatura
    ax = sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
    ax.set(xlabel='Variables', ylabel='Número de cluster')
    bottom, top = ax.get_ylim()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.savefig("./fig/caso2/" + name + "_heatmap.png")
    if show:
        plt.show()
    plt.clf() 
# Heatmap '''

#'''Scatter matrix
def scatter_matrix(X_alg, show=False):
    print("---------- Preparando el scatter matrix " + name + "...")

    #se añade la asignación de clusters como columna a X
    sns.set()
    variables = list(X_alg)
    variables.remove('cluster')
    sns_plot = sns.pairplot(X_alg, vars=variables, hue="cluster", palette='Paired', plot_kws={"s": 25}, diag_kind="hist") 
    plt.subplots_adjust(wspace=.03, hspace=.03)
    plt.savefig("./fig/caso2/" + name + "_scattermatrix.png")
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
            
    plt.savefig("./fig/caso2/" + name + "_distplot.png")
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

    fig.savefig("./fig/caso2/" + name + "_boxplot.png")
    if show:
        plt.show()
    plt.clf()
#'''

#---------------------------------------------------------
    
# Leemos el csv desde el propio directorio
censo = pd.read_csv('mujeres_fecundidad_INE_2018.csv')

# Imputaremos los valores perdidos con la media
prueba = pd.DataFrame(censo)


#bar_graph(censo, 'M_NOHIJOS1')

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
# Instanciación del algoritmo, indico nº de clusters
# Algoritmo k-means
k_means = KMeans(init='k-means++', n_clusters=5, n_init=5)

# Algoritmo MeanShift
# Parámetro: radio de los clusters, por defecto utiliza  sklearn.cluster.estimate_bandwidth
ms = MeanShift()

# Algoritmo Birch
# Parámetros: factor de ramificación, umbral, número de clusters (=None devuelve los que haya generado)
birch = Birch(branching_factor=25, threshold=0.25, n_clusters=5)

# Algoritmo DBSCAN
# Parámetros: eps (mínima distancia), min_samples (número de ejemplos en el vecindario de un punto para considerarlo centro)
dbscan = DBSCAN(eps=0.2, min_samples=5)

# Algoritmo Ward
#Vamos a usar este jerárquico y nos quedamos con 100 clusters, es decir, cien ramificaciones del dendrograma
ward = AgglomerativeClustering(n_clusters=5, linkage='ward')

# Lista de algoritmos a utilizar
algorithms = (('KMeans', k_means), ('MeanShift', ms), ('Birch', birch), ('DBSCAN', dbscan), ('Ward', ward)) 
cluster_predict = {}

# Variable de centros a utilizar
centroids = {'KMeans':"cluster_centers_", 'MeanShift':"cluster_centers_", 'Birch':"subcluster_centers_"}

# Bucle algoritmos
filas_tabla_res = []
for name, alg in algorithms:
    print('----- Ejecutando ' + name, end='') # -----
    
    #Tomamos tiempos
    t = time.time()
    # Ejecuto el algoritmo y asigno los clusters
    cluster_predict[name] = alg.fit_predict(X_normal)
    tiempo = time.time() - t
    #Pinto resultados
    print(": {:.2f} segundos, ".format(tiempo), end='')
    try:
        metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict[name])
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
        metric_SC = metrics.silhouette_score(X_normal, cluster_predict[name], metric='euclidean', sample_size = floor(muestra_silhoutte*len(X)), random_state=123456)
        print("Silhouette Coefficient: {:.5f}".format(metric_SC))
    except:
        print("No podemos calcular el índice silhouette")
        metric_SC = -1
        
    # Calculamos el número de clusters (será interesante en los algoritmos que no lo imponemos)
    labels = alg.labels_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("Número de clusters: {:.0f}".format(n_clusters_))


    # Fila tabla
    s = name + " & "
    s += "{:.3f}".format(tiempo) + " & "
    s += "{:.3f}".format(metric_CH) + " & "
    s += "{:.5f}".format(metric_SC) + " & "
    s += "{:.0f}".format(n_clusters_) + " \\\\"
    
    filas_tabla_res.append(s)

    # Visualización
    clusters = pd.DataFrame(cluster_predict[name],index=X.index,columns=['cluster'])
    X_alg = pd.concat([X, clusters], axis=1)
    tam_clusters(clusters)

    if name != 'Ward':
        try:
            scatter_matrix(X_alg)
        except:
          print("------ERROR: No se pudo crear el scatter matrix------")
        try:
            distplot(X_alg, name, n_clusters_, usadas)
        except:
            print("------ERROR: No se pudo crear el distplot-----")
        try:
            boxplot(X_alg, name, n_clusters_, usadas)
        except:
            print("------ERROR: No se pudo crear el boxplot------")
    # Solo se puede generar para los algoritmos que generan clusters convexos
    if name in centroids:
        centers = pd.DataFrame(getattr(alg, centroids[name]) ,columns=list(X))
        print(name + "/n", centers)
    
        heatmap(centers)
    #----

tabla_alg(filas_tabla_res)

############ Jerárquico
#Filtro quitando los elementos (outliers) que caen en clusters muy pequeños en el jerárquico
#'''
min_size = 10
k_ward = len(set(cluster_predict['Ward']))
X_filtrado = X_alg[X_alg.groupby('cluster').cluster.transform(len) > min_size]
k_filtrado = len(set(X_filtrado['cluster']))
print("De los {:.0f} clusters hay {:.0f} con más de {:.0f} elementos. Del total de {:.0f} elementos, se seleccionan {:.0f}".format(k_ward,k_filtrado,min_size,len(X),len(X_filtrado)))
X_filtrado = X_filtrado.drop('cluster', 1)
#'''

X_filtrado = X

#Normalizo el conjunto filtrado
X_filtrado_normal = preprocessing.normalize(X_filtrado, norm='l2')

#Saco el dendrograma usando scipy, que realmente vuelve a ejecutar el clustering jerárquico
linkage_array = hierarchy.ward(X_filtrado_normal)
plt.figure(1)
plt.clf()
dendro = hierarchy.dendrogram(linkage_array,orientation='left') #lo pongo en horizontal para compararlo con el generado por seaborn
#puedo usar, por ejemplo, "p=10,truncate_mode='lastp'" para cortar el dendrograma en 10 hojas
plt.savefig("./fig/caso2/Ward_dendrograma.png")


X_filtrado_normal_DF = pd.DataFrame(X_filtrado_normal,index=X_filtrado.index,columns=usadas)
sns.set()
sns_dendro = sns.clustermap(X_filtrado_normal_DF, method='ward', col_cluster=False, figsize=(20,10), cmap="YlGnBu", yticklabels=False)

sns_dendro.savefig("./fig/caso2/Ward_dendro_heat.png")
