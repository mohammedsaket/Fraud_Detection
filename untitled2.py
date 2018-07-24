# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 01:26:57 2018

@author: MOHMMED SAKET
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score,precision_score
df = pd.read_csv('fraud_data.csv')
#print(df)

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.dummy import DummyClassifier
dc = DummyClassifier(random_state = 0)
svc = SVC(C=1e9,gamma=1e-07)
lr = LogisticRegression()
for i in [dc,svc,lr]:
    i.fit(X_train,y_train)
    print('accuracy=',(i.score(X_test,y_test))*100,'%')
import seaborn as sns
import matplotlib.pyplot as plt
numeric_data_select = df[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28']]
corr_select = numeric_data_select.corr()
plt.figure(figsize=(8, 8))
mask = np.zeros_like(corr_select)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_select, vmax=1, square=True, annot=True, mask=mask, cbar=False, linewidths=0.1)
plt.xticks(rotation=45)
sns.pairplot(numeric_data_select, size=2)
