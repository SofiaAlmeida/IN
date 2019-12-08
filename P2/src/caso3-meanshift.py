# -*- coding: utf-8 -*-
"""
Autor:
    Sofía Almeida Bruno
Basada en el código proporcionado por:
    Jorge Casillas
Fecha:
    Noviembre/2019
"""
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.cluster import MeanShift, estimate_bandwidth

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
    print("\caption{Resultados cambio de parámetros MeanShift}")
    print("\label{tab:param_meanshift3}")
    print("\\begin{tabular}{lrrrr}")
    print("\\toprule")
    print("bandwidth & Tiempo ($s$) & Calinski-Harabasz & Silhouette & Número de clusters\\\\")
    print("\midrule")

    # Contenido de la tabla
    for s in l:
        print(s)
    
    # Cierre de la tabla
    print("\\bottomrule")
    print("\end{tabular}")
    print("\end{table}")
    
#'''Gráfica lineal
def line_graph(param_data, etiq_x, etiq_y, etiq_y2, show=False):
    print("--------Creando gráfico lineal-----")
    sns.set()
    fig, ax =plt.subplots(1,2, figsize=(15,10))
    fig.subplots_adjust(wspace=0.2)
    
    sns.lineplot(x=etiq_x, y=etiq_y, data=param_data, ax=ax[0])
    sns.lineplot(x=etiq_x, y=etiq_y2, data=param_data, ax=ax[1])
    ax[0].set(xlabel='Radio')
    ax[1].set(xlabel='Radio')
    
    fig.savefig("./fig/caso3/param_ms.png")

    if show:
        plt.show()

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


print("me dispongo a estimar el radio...")
initial_bandwidth = estimate_bandwidth(subset)
print("bandwidth estimado:", initial_bandwidth)

# seleccionar variables de interés para clustering
usadas = ['EDADTRAREPRO', 'NTRABA', 'TEMPRELA', 'TDYO']
X = subset[usadas] # Columnas que voy a estudiar

#Normalizamos
X_normal = X.apply(norm_to_zero_one)

#Ejecuto clustering

# Bucle parámetros
filas_tabla_res = []
sil = []
ch = []
# band_width = [(initial_bandwidth / i)  for i in range(12, 22)]
# band_width = [(initial_bandwidth * i)  for i in range(1, 11)]
band_width = [(initial_bandwidth / i)  for i in range(90, 100)]

print(band_width)
for bw in band_width:
    print('----- Ejecutando MeanShift, bandwidth: ' + str(bw), end='') # -----
    
    #Tomamos tiempos
    t = time.time()
    # Ejecuto el algoritmo y asigno los clusters
    # fit: construir clusters
    # predict: contiene una columna con el cluster asignado a cada objeto
    ms = MeanShift(bandwidth=100, cluster_all=True)
    

    cluster_predict = ms.fit_predict(X_normal)
    tiempo = time.time() - t
    #Pinto resultados
    print(": {:.2f} segundos, ".format(tiempo), end='')
    try:
        metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict)
        print("Calinski-Harabasz Index: {:.3f}, ".format(metric_CH), end='')
        ch.append(metric_CH)
    except:
        print("----ERROR: No podemos calcular el índice Calinski-Harabasz---")
        metric_CH = -1
    # Otra medida de rendimiento, menos eficiente
    #el cálculo de Silhouette puede consumir mucha RAM. Si son muchos datos, digamos más de 10k, se puede seleccionar una muestra, p.ej., el 20%
    if len(X) > 10000:
       muestra_silhoutte = 0.2
    else:
       muestra_silhoutte = 1.0
    
    # Para hacer pruebas podemos hacer un muestreo, para los resultados definitivos ya dejamos que se ejecute   
    try:
        metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', sample_size = floor(muestra_silhoutte*len(X)), random_state=123456)
        print("Silhouette Coefficient: {:.5f}".format(metric_SC))
        sil.append(metric_SC)
    except:
        print("No podemos calcular el índice silhouette")
        metric_SC = -1
    # Calculamos el número de clusters (será interesante en los algoritmos que no lo imponemos)
    labels = ms.labels_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("Número de clusters: {:.0f}".format(n_clusters_))
    

    # Fila tabla
    s = "{:.0f}".format(bw)  + " & "
    s += "{:.3f}".format(tiempo) + " & "
    s += "{:.3f}".format(metric_CH) + " & "
    s += "{:.5f}".format(metric_SC) + " & "
    s += "{:.0f}".format(n_clusters_) + " \\\\"
    
    filas_tabla_res.append(s)
    #----
    # Visualización
d = {'bandwidth' : band_width, 'Silhouette' : sil, 'Calinski-Harabasz' : ch}
line_graph(pd.DataFrame(d), "bandwidth", "Calinski-Harabasz", "Silhouette")

tabla_alg(filas_tabla_res)

