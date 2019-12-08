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
    plt.savefig("./fig/caso3/ward_heatmap.png")
    if show:
        plt.show()
    plt.clf() 
# Heatmap '''

#---------------------------------------------------------
    
# Leemos el csv desde el propio directorio
censo = pd.read_csv('mujeres_fecundidad_INE_2018.csv')

# Imputaremos los valores perdidos con la media
prueba = pd.DataFrame(censo)

for col in censo:
   censo[col].fillna(censo[col].mean(), inplace=True)

#seleccionar casos
subset = censo.loc[(censo['EMBANT'] == 1) & (censo['TRAREPRO'] == 1)]
   
print("Subconjunto formado por " + str(len(subset)) + " objetos")
# seleccionar variables de interés para clustering
usadas = ['EDADTRAREPRO', 'NTRABA', 'TEMPRELA', 'TDYO']
X = subset[usadas] # Columnas que voy a estudiar

#Normalizamos
X_normal = X.apply(norm_to_zero_one)

# Ejecuto clustering
# Algoritmo Ward
#Vamos a usar este jerárquico y nos quedamos con 100 clusters, es decir, cien ramificaciones del dendrograma
ward = AgglomerativeClustering(n_clusters=5, linkage='ward')

# Lista de algoritmos a utilizar
cluster_predict = {}

# Bucle algoritmos
filas_tabla_res = []
print('----- Ejecutando ' + 'Ward', end='') # -----
    
#Tomamos tiempos
t = time.time()
# Ejecuto el algoritmo y asigno los clusters
cluster_predict['Ward'] = ward.fit_predict(X_normal)
tiempo = time.time() - t
#Pinto resultados
print(": {:.2f} segundos, ".format(tiempo), end='')
try:
    metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict['Ward'])
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
    metric_SC = metrics.silhouette_score(X_normal, cluster_predict['Ward'], metric='euclidean', sample_size = floor(muestra_silhoutte*len(X)), random_state=123456)
    print("Silhouette Coefficient: {:.5f}".format(metric_SC))
except:
    print("No podemos calcular el índice silhouette")
    metric_SC = -1
        
# Calculamos el número de clusters (será interesante en los algoritmos que no lo imponemos)
labels = ward.labels_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)
print("Número de clusters: {:.0f}".format(n_clusters_))

# Fila tabla
s = 'Ward' + " & "
s += "{:.3f}".format(tiempo) + " & "
s += "{:.3f}".format(metric_CH) + " & "
s += "{:.5f}".format(metric_SC) + " & "
s += "{:.0f}".format(n_clusters_) + " \\\\"
    
filas_tabla_res.append(s)

# Visualización
clusters = pd.DataFrame(cluster_predict['Ward'],index=X.index,columns=['cluster'])
X_alg = pd.concat([X, clusters], axis=1)

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
plt.clf()
sns.set()
dendro = hierarchy.dendrogram(linkage_array,orientation='left', p=10, truncate_mode='lastp') #lo pongo en horizontal para compararlo con el generado por seaborn
#puedo usar, por ejemplo, "p=10,truncate_mode='lastp'" para cortar el dendrograma en 10 hojas

plt.savefig("./fig/caso3/Ward_dendrograma3.png")


X_filtrado_normal_DF = pd.DataFrame(X_filtrado_normal,index=X_filtrado.index,columns=usadas)


filtrado_pal = sns.cubehelix_palette(n_clusters_,
                                    light=.9, dark=.1, reverse=True,
                                    start=1, rot=-2)
filtrado_lut = dict(zip(map(str, labels_unique), filtrado_pal))

filtrado_colors = pd.Series(labels_unique).map(filtrado_lut)


sns_dendro = sns.clustermap(X_filtrado_normal_DF, method='ward', col_cluster=False, row_colors=filtrado_colors, figsize=(20,10), cmap="YlGnBu", yticklabels=False)

sns_dendro.savefig("./fig/caso3/Ward_dendro_heat3.png")
