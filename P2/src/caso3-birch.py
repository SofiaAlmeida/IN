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

from sklearn.cluster import Birch

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
    print("\caption{Resultados cambio de parámetros Birch}")
    print("\label{tab:param_birch3}")
    print("\\begin{tabular}{llrrrrr}")
    print("\\toprule")
    print(" Umbral & Factor de ramificación & Tiempo ($s$) & Calinski-Harabasz & Silhouette & Número de clusters \\\\")
    print("\midrule")

    # Contenido de la tabla
    for s in l:
        print(s)
    
    # Cierre de la tabla
    print("\\bottomrule")
    print("\end{tabular}")
    print("\end{table}")
    
#'''Gráfica lineal
def line_graph(param_data, etiq_x, etiq_y, etiq_y2, n_samples, show=False):
    print("--------Creando gráfico lineal-----")
    sns.set()
    fig, ax =plt.subplots(1,2, figsize=(15,10))
    fig.subplots_adjust(wspace=0.2)
    
    sns.lineplot(x=etiq_x, y=etiq_y, data=param_data, ax=ax[0])
    sns.lineplot(x=etiq_x, y=etiq_y2, data=param_data, ax=ax[1])
    ax[0].set(xlabel='')
    ax[1].set(xlabel='')
    fig.set_title=('n_samples =' + str(n_samples))
    
    fig.savefig("./fig/caso3/param_birch" + str(n_samples) + ".png")

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
branching_factor = [15, 20, 25, 30, 35]
threshold = [0.1, 0.15, 0.2, 0.25, 0.3]
for b in branching_factor:
    for th in threshold:
        print('----- Ejecutando Birch, branching factor: ' + str(b) + ', threshold: ' + str(th), end='') # -----
    
        #Tomamos tiempos
        t = time.time()
        # Ejecuto el algoritmo y asigno los clusters
        birch = Birch(branching_factor=b, threshold=th, n_clusters=5)
    

        cluster_predict = birch.fit_predict(X_normal)
        tiempo = time.time() - t
        #Pinto resultados
        print(": {:.2f} segundos, ".format(tiempo), end='')
        try:
            metric_CH = metrics.calinski_harabasz_score(X_normal, cluster_predict)
            print("Calinski-Harabasz Index: {:.3f}, ".format(metric_CH), end='')
        except:
            print("----ERROR: No podemos calcular el índice Calinski-Harabasz---")
            metric_CH = -1

        ch.append(metric_CH)
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
        except:
            print("ERROR : No podemos calcular el índice silhouette")
            metric_SC = -1
        sil.append(metric_SC)
            
        # Calculamos el número de clusters (será interesante en los algoritmos que no lo imponemos)
        labels = birch.labels_

        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        print("Número de clusters: {:.0f}".format(n_clusters_))

        tams = []
        clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])
        print("Tamaño de cada cluster:")
        size=clusters['cluster'].value_counts()
        for num,i in size.iteritems():
            print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
            tams.append(100*i/len(clusters))
        # Fila tabla
        s = "{:.2f}".format(th)  + " & "
        s += "{:.0f}".format(b) + " & "
        s += "{:.3f}".format(tiempo) + " & "
        s += "{:.3f}".format(metric_CH) + " & "
        s += "{:.5f}".format(metric_SC) + " & "
        s += "{:.0f}".format(n_clusters_) + " \\\\"
    
        filas_tabla_res.append(s)
    #----
    # Visualización
    print(len(branching_factor), len(sil), len(ch))
    d = {'branching_factor' : branching_factor, 'Silhouette' : sil, 'Calinski-Harabasz' : ch}
    line_graph(pd.DataFrame(d), "branching_factor", "Calinski-Harabasz", "Silhouette", th)
    sil.clear()
    ch.clear()
    
tabla_alg(filas_tabla_res)

