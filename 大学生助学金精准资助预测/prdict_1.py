# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 18:24:22 2016
预测
@author: 80374769
"""

import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

tr_src = pd.read_csv(r'E:\help\train.csv')
te = pd.read_csv(r'E:\help\test.csv')

del tr_src['stu_id']




oversampling_1 = tr_src[tr_src['class']==1]
oversampling_2 = tr_src[tr_src['class']==2]
oversampling_3 = tr_src[tr_src['class']==3]
sample_0 = tr_src[tr_src['class']==0]

tr = tr_src.append(sample_0)

for i in range(len(sample_0)/len(oversampling_1)-3):
    tr = tr.append(oversampling_1)
for i in range(len(sample_0)/len(oversampling_2)-3):
    tr = tr.append(oversampling_2)    
for i in range(len(sample_0)/len(oversampling_3)-3):
    tr = tr.append(oversampling_3)
tr.index = [i for i in range(len(tr))]  
inde = list(tr.index)
random.shuffle(inde)
tr = tr.ix[inde]

feature_list = list(tr.columns)[1:]
tr_x = tr[feature_list]
tr_y = tr['class']

te_x = te[feature_list]
te_x = te_x.fillna(0)
from sklearn import preprocessing

X = pd.concat([tr_x,te_x],ignore_index=True)
X = preprocessing.scale(X)
tr_x_nor = X[:len(tr_x)]
te_x_nor = X[len(tr_x):]

clf = RandomForestClassifier(random_state=123)
clf = clf.fit(tr_x_nor,tr_y)
result = clf.predict(te_x_nor)
print len(result!=0)

clf_gb = GradientBoostingClassifier(random_state=2016)
clf_gb = clf_gb.fit(tr_x_nor,tr_y)
result_gb = clf_gb.predict(te_x_nor)
print len(result_gb!=0)

y =[]
for i in result:
    if i ==0:
        y.append(0)
    elif i == 1:
        y.append(1000)
    elif i == 2:
        y.append(1500)
    elif i ==3:
        y.append(2000)
    else:
        print y
id = te.stu_id
to_save = pd.DataFrame({'id':id,'zz':y})
to_save.to_csv(r're.csv',index=False)