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
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import make_scorer
from sklearn.feature_selection import VarianceThreshold

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def plotImp(model, features , num = 20):
    feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':features})
    plt.figure(figsize=(40, 20))
    sns.set(font_scale = 3.75)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 
                                                        ascending=False)[0:num])
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('../fig/lgbm_importances' + sys.argv[0][-5:-3] + '.png')
    
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

def preprocessing(data_x, data_x_tst):
    print("----- Preprocessing...")
    print("Nº variables inicial: " + str(len(data_x.columns)))

    categorical_feature_mask = data_x.dtypes==object
    # Filtramos las columnas categóricas usando la máscara y la convertimos en una lista
    categorical_cols = data_x.columns[categorical_feature_mask].tolist()

    le = LabelEncoder()
    data_x[categorical_cols] = data_x[categorical_cols].apply(lambda col: le.fit_transform(col))
    ohe = OneHotEncoder(categorical_features = categorical_feature_mask, sparse = False)
    data_x_tmp = ohe.fit_transform(data_x)
    print("Nº variables tras OneHotEncoder: " + str(data_x_tmp.shape[1]))
    sel = VarianceThreshold(threshold=(.95 * (1 - .95)))
    X = sel.fit_transform(data_x_tmp)

    print("Nº variables final: " + str(X.shape[1]))

    data_x_tst[categorical_cols] = data_x[categorical_cols].apply(lambda col: le.fit_transform(col))
    data_x_tmp = ohe.transform(data_x_tst)
    X_tst = sel.fit_transform(data_x_tmp)
    return X, X_tst
#------------------------------------------------------------------

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

X, X_tst = preprocessing(data_x, data_x_tst)
y = np.ravel(data_y.values)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123456)
    
print("------ LightGBM...")
lgbm = lgb.LGBMClassifier(objective='regression_l1', n_estimators=200, n_jobs=2, num_leaves = 45, scale_pos_weight = 0.1)
lgbm, y_test_lgbm = validacion_cruzada(lgbm, X, y, skf)


# Entreno de nuevo con el total de los datos
# El resultado que muestro es en training, será mejor que en test
clf = lgbm
clf = clf.fit(X,y)
#plotImp(clf, selec, X.shape[1])
y_pred_tra = clf.predict(X)
print("F1 score (tra): {:.4f}".format(f1_score(y,y_pred_tra,average='micro')))

y_pred_tst = clf.predict(X_tst)

df_submission['damage_grade'] = y_pred_tst
df_submission.to_csv("../Submissions/submission_" + sys.argv[0][-5:-3] + ".csv", index=False)
