# -*- coding: utf-8 -*-
"""
Autora:
    Sofía Almeida
Fecha:
    Diciembre/2019
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import lightgbm as lgb
import sys
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


'''
Validación cruzada con particionado estratificado y control de la aleatoriedad fijando la semilla
'''
def validacion_cruzada(modelo, X, y, cv):
    print("------ Validación cruzada...")
    y_test_all = []

    for train, test in cv.split(X, y):
        t = time.time()
        modelo = modelo.fit(X[train],y[train])
        tiempo = time.time() - t
        y_pred = modelo.predict(X[test])
        print("F1 score (tst): {:.4f}, tiempo: {:6.2f} segundos".format(f1_score(y[test],y_pred,average='micro') , tiempo))
        y_test_all = np.concatenate([y_test_all,y[test]])

    print("")

    return modelo, y_test_all
#------------------------------------------------------------------------


'''Ajuste de parámetros
# n = n_estimators'''
def ajuste_lgbm(n, X, y, params_lgbm):
    print("------ Grid Search...")
    grid = GridSearchCV(lgbm, params_lgbm, cv=3, n_jobs=1, verbose=1, scoring=make_scorer(f1_score, average='micro'))
    grid.fit(X, y)
    print(grid.cv_results_)
    print("Mejores parámetros:")
    print(grid.best_params_)
    return grid.best_estimator_
#------------------------------------------------------------------


'''lightgbm más reciente, más eficiente que xgb (que igual tarda 2-3 min)
lightgbm no suele conseguir más precisión, gana en tiempo
Como para exprimir el algoritmo hacemos búsqueda de parámetros...
el tiempo es fundamental
#'''
le = preprocessing.LabelEncoder()

'''
lectura de datos
'''
print("----- Leyendo datos ...")
#los ficheros .csv se han preparado previamente para sustituir ,, y "Not known" por NaN (valores perdidos)
data_x = pd.read_csv('../data/nepal_earthquake_tra.csv')
data_y = pd.read_csv('../data/nepal_earthquake_labels.csv')
data_x_tst = pd.read_csv('../data/nepal_earthquake_tst.csv')
df_submission = pd.read_csv('../data/nepal_earthquake_submission_format.csv')

#se quitan las columnas que no se usan
data_x.drop(labels=['building_id'], axis=1,inplace = True)
data_x_tst.drop(labels=['building_id'], axis=1,inplace = True)
data_y.drop(labels=['building_id'], axis=1,inplace = True)

def preprocessing(data_x, data_x_tst):
    '''
Se convierten las variables categóricas a variables numéricas (ordinales)
'''
    print("----- Preprocessing...")

    print("Nº variables inicial: " + str(len(data_x.columns)))
    data_x_tmp = pd.get_dummies(data=data_x, columns=['land_surface_condition', 'foundation_type',  'roof_type', 'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status'])
    X = data_x_tmp.values
    print("Nº variables final: " + str(X.shape[1]))
    
    data_x_tmp = pd.get_dummies(data=data_x_tst, columns=['land_surface_condition', 'foundation_type',  'roof_type', 'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status'])
    X_tst = data_x_tmp.values
    return X, X_tst

X, X_tst = preprocessing(data_x, data_x_tst)
y = np.ravel(data_y.values)

#------------------------------------------------------------------------

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)
    
print("------ LightGBM...")
lgbm = lgb.LGBMClassifier(objective='regression_l1', n_estimators=200, n_jobs=2, num_leaves = 40, scale_pos_weight = 0.1)
lgbm, y_test_lgbm = validacion_cruzada(lgbm, X, y, skf)



# Entreno de nuevo con el total de los datos
# El resultado que muestro es en training, será mejor que en test
clf = lgbm
clf = clf.fit(X,y)
y_pred_tra = clf.predict(X)
print("F1 score (tra): {:.4f}".format(f1_score(y,y_pred_tra,average='micro')))
y_pred_tst = clf.predict(X_tst)

df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("../Submissions/submission_" + sys.argv[0][-5:-3] + ".csv", index=False)
