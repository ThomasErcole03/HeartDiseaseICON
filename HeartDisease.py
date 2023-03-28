# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 19:04:16 2023

@author: Thomas Ercole
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
import xgboost as xgb
from pgmpy.estimators import K2Score, HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.svm import SVC

df = pd.read_csv('heart.csv')
print(df.head(10))


# Nomino le colonne del dataset
cols = ["age","gender","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]

# K-FOLD e classificazione

# Creazione features X e target y
X = df.to_numpy()
y = df["target"].to_numpy()

# K-Fold Cross Validation
kf = KFold(n_splits=10, shuffle=True, random_state= 0)

# Classificatori per la valutazione
knn = KNeighborsClassifier(weights="distance")
dtc = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0,max_depth=3)
histg= HistGradientBoostingClassifier(random_state=0)
adaboost= AdaBoostClassifier(random_state=0,learning_rate=0.8)
svc = SVC(random_state=0)
xgboost= xgb.XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=3)

model = {
    'KNN': {'accuracy_list': [],
            'precision_list': [],
            'recall_list': [],
            'f1_list': []
            },

    'DecisionTree': {'accuracy_list': [],
                     'precision_list': [],
                     'recall_list': [],
                     'f1_list': []
                     },

    'RandomForest': {'accuracy_list': [],
                     'precision_list': [],
                     'recall_list': [],
                     'f1_list': []
                     },
    'HistGradientBoosting' :{
                     'accuracy_list': [],
                     'precision_list': [],
                     'recall_list': [],
                     'f1_list': []
                      },    

    'AdaBoost' :{
                     'accuracy_list': [],
                     'precision_list': [],
                     'recall_list': [],
                     'f1_list': []
                      },    

    'SVC': {'accuracy_list': [],
            'precision_list': [],
            'recall_list': [],
            'f1_list': []
            },
         
    'ExtremeGradientBoosting' :{
                     'accuracy_list': [],
                     'precision_list': [],
                     'recall_list': [],
                     'f1_list': []
                      }   


}


# K-Fold dei classificatori
for train_index, test_index in kf.split(X, y):
    training_set, testing_set = X[train_index], X[test_index]

    # Dati di train
    data_train = pd.DataFrame(training_set, columns=df.columns)
    X_train = data_train.drop("target", axis=1)
    y_train = data_train.target

    # Dati di test
    data_test = pd.DataFrame(testing_set, columns=df.columns)
    X_test = data_test.drop("target", axis=1)
    y_test = data_test.target

    # Fit dei classificatori
    knn.fit(X_train, y_train)
    dtc.fit(X_train, y_train)
    rfc.fit(X_train, y_train)
    adaboost.fit(X_train, y_train)
    histg.fit(X_train, y_train)
    svc.fit(X_train, y_train)
    xgboost.fit(X_train, y_train)

    y_pred_knn = knn.predict(X_test)
    y_pred_dtc = dtc.predict(X_test)
    y_pred_rfc = rfc.predict(X_test)
    y_pred_histg=adaboost.predict(X_test)
    y_pred_adab= histg.predict(X_test)
    y_pred_svc = svc.predict(X_test)
    y_pred_xgboost=xgboost.predict(X_test)
    # Salvo le metriche del fold nel dizionario per i classificatori
    model['KNN']['accuracy_list'].append(metrics.accuracy_score(y_test, y_pred_knn))
    model['KNN']['precision_list'].append(metrics.precision_score(y_test, y_pred_knn))
    model['KNN']['recall_list'].append(metrics.recall_score(y_test, y_pred_knn))
    model['KNN']['f1_list'].append(metrics.f1_score(y_test, y_pred_knn))

    model['DecisionTree']['accuracy_list'].append(metrics.accuracy_score(y_test, y_pred_dtc))
    model['DecisionTree']['precision_list'].append(metrics.precision_score(y_test, y_pred_dtc))
    model['DecisionTree']['recall_list'].append(metrics.recall_score(y_test, y_pred_dtc))
    model['DecisionTree']['f1_list'].append(metrics.f1_score(y_test, y_pred_knn))

    model['RandomForest']['accuracy_list'].append(metrics.accuracy_score(y_test, y_pred_rfc))
    model['RandomForest']['precision_list'].append(metrics.precision_score(y_test, y_pred_rfc))
    model['RandomForest']['recall_list'].append(metrics.recall_score(y_test, y_pred_rfc))
    model['RandomForest']['f1_list'].append(metrics.f1_score(y_test, y_pred_rfc))

    model['HistGradientBoosting']['accuracy_list'].append(metrics.accuracy_score(y_test, y_pred_histg))
    model['HistGradientBoosting']['precision_list'].append(metrics.precision_score(y_test, y_pred_histg))
    model['HistGradientBoosting']['recall_list'].append(metrics.recall_score(y_test, y_pred_histg))
    model['HistGradientBoosting']['f1_list'].append(metrics.f1_score(y_test, y_pred_histg))

    model['AdaBoost']['accuracy_list'].append(metrics.accuracy_score(y_test, y_pred_adab))
    model['AdaBoost']['precision_list'].append(metrics.precision_score(y_test, y_pred_adab))
    model['AdaBoost']['recall_list'].append(metrics.recall_score(y_test, y_pred_adab))
    model['AdaBoost']['f1_list'].append(metrics.f1_score(y_test, y_pred_adab))

    model['ExtremeGradientBoosting']['accuracy_list'].append(metrics.accuracy_score(y_test, y_pred_xgboost))
    model['ExtremeGradientBoosting']['precision_list'].append(metrics.precision_score(y_test, y_pred_xgboost))
    model['ExtremeGradientBoosting']['recall_list'].append(metrics.recall_score(y_test, y_pred_xgboost))
    model['ExtremeGradientBoosting']['f1_list'].append(metrics.f1_score(y_test, y_pred_xgboost))

    model['SVC']['accuracy_list'].append(metrics.accuracy_score(y_test, y_pred_svc))
    model['SVC']['precision_list'].append(metrics.precision_score(y_test, y_pred_svc))
    model['SVC']['recall_list'].append(metrics.recall_score(y_test, y_pred_svc))
    model['SVC']['f1_list'].append(metrics.f1_score(y_test, y_pred_svc))


# Media delle metriche del KNN
print("\nMedia delle metriche del KNN")
print("Media Accuracy: %f" % (np.mean(model['KNN']['accuracy_list'])))
print("Media Precision: %f" % (np.mean(model['KNN']['precision_list'])))
print("Media Recall: %f" % (np.mean(model['KNN']['recall_list'])))
print("Media f1: %f" % (np.mean(model['KNN']['f1_list'])))

# Media delle metriche del DecisionTree
print("\nMedia delle metriche del DecisionTree")
print("Media Accuracy: %f" % (np.mean(model['DecisionTree']['accuracy_list'])))
print("Media Precision: %f" % (np.mean(model['DecisionTree']['precision_list'])))
print("Media Recall: %f" % (np.mean(model['DecisionTree']['recall_list'])))
print("Media f1: %f" % (np.mean(model['DecisionTree']['f1_list'])))


# Media delle metriche della RandomForest
print("\nMedia delle metriche del RandomForest")
print("Media Accuracy: %f" % (np.mean(model['RandomForest']['accuracy_list'])))
print("Media Precision: %f" % (np.mean(model['RandomForest']['precision_list'])))
print("Media Recall: %f" % (np.mean(model['RandomForest']['recall_list'])))
print("Media f1: %f" % (np.mean(model['RandomForest']['f1_list'])))

# Media delle metriche della HistgradientBoosting
print("\nMedia delle metriche del HistgradientBoosting")
print("Media Accuracy: %f" % (np.mean(model['HistGradientBoosting']['accuracy_list'])))
print("Media Precision: %f" % (np.mean(model['HistGradientBoosting']['precision_list'])))
print("Media Recall: %f" % (np.mean(model['HistGradientBoosting']['recall_list'])))
print("Media f1: %f" % (np.mean(model['HistGradientBoosting']['f1_list'])))

# Media delle metriche della AdaBoost
print("\nMedia delle metriche di AdaBoost")
print("Media Accuracy: %f" % (np.mean(model['AdaBoost']['accuracy_list'])))
print("Media Precision: %f" % (np.mean(model['AdaBoost']['precision_list'])))
print("Media Recall: %f" % (np.mean(model['AdaBoost']['recall_list'])))
print("Media f1: %f" % (np.mean(model['AdaBoost']['f1_list'])))

# Media delle metriche del SVM
print("\nMedia delle metriche del SVM")
print("Media Accuracy: %f" % (np.mean(model['SVC']['accuracy_list'])))
print("Media Precision: %f" % (np.mean(model['SVC']['precision_list'])))
print("Media Recall: %f" % (np.mean(model['SVC']['recall_list'])))
print("Media f1: %f" % (np.mean(model['SVC']['f1_list'])))

# Media delle metriche del Xgb
print("\nMedia delle metriche del ExtremeGradientBoosting")
print("Media Accuracy: %f" % (np.mean(model['ExtremeGradientBoosting']['accuracy_list'])))
print("Media Precision: %f" % (np.mean(model['ExtremeGradientBoosting']['precision_list'])))
print("Media Recall: %f" % (np.mean(model['ExtremeGradientBoosting']['recall_list'])))
print("Media f1: %f" % (np.mean(model['ExtremeGradientBoosting']['f1_list'])))    


# Rete Bayesiana

# Conversione dei valori all'interno del dataframe in interi
df_int = np.array(df, dtype=int)
df = pd.DataFrame(df_int, columns=df.columns)

# Creazione struttura di rete
k2 = K2Score(df)
hc_k2 = HillClimbSearch(df)
k2_model = hc_k2.estimate(scoring_method=k2)

# Creazione della rete bayesiana e fit
bNet = BayesianNetwork(k2_model.edges())
bNet.fit(data=df, estimator=MaximumLikelihoodEstimator)

# Eliminazione variabili ininfluenti
data = VariableElimination(bNet)

# Definiamo evidenze da dare in input a metodi di Inferenza
print("Seleziona una o più di queste variabili (in caso di multiple, separale con uno spazio):")
for s in cols:
    print(s)
_input = input(" > ")
var_list = _input.split(" ")
for v in var_list:
    if v not in cols:
        raise Exception("Variabile selezionata non presente nella lista")
for x in var_list:
    cols.remove(x)
print("Seleziona l'evidenza dalla lista specificando il valore (ad esempio, gender:1 oldpeak:1 fbs:1):")
for s in cols:
    print(s)
_input = input(" > ")
evidence_list = _input.split(" ")
tokens = []
for e in evidence_list:
    tokens += e.split(":")
       
evidence_dict = {}
if evidence_list != ['']:
    for t in tokens:
        if not t.isnumeric() and t not in cols:
            raise Exception("Variabile selezionata non presente nella lista")
it = iter(tokens)
for x in it:
    evidence_dict.update({ x : int(next(it))})

# Query definita e stampa risultato
result = data.query(variables=var_list, evidence=evidence_dict)
print(result)
