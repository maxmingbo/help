# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 23:31:26 2016

@author: zmb
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 16:58:12 2016

@author: 80374769
"""

import pandas as pd
import numpy as np

from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

train_test = pd.read_csv(r'input\train_test.csv')

train = train_test[train_test['money'].notnull()]
test = train_test[train_test['money'].isnull()]
                  
train.college = train.fillna(method='pad')['college']
train.score = train.fillna(method='pad')['score']
train.num = train.fillna(method='pad')['num']
train.order = train.fillna(method='pad')['order']

#train = train.dropna()

test.score = test.fillna(method='pad')['score']
test.num = test.fillna(method='pad')['num']
test.order = test.fillna(method='pad')['order']
test.college = test.fillna(method='pad')['college']

train = train.fillna(0)
test = test.fillna(0)


target = 'money'
IDcol = 'id'
ids = test['id'].values
predictors = [x for x in train.columns if x not in [target,'id']]


t_x, val_x, t_y, y_true = train_test_split(train[predictors], train[target], test_size=0.25, random_state=0)  

train_temp = pd.concat([t_x,t_y],axis = 1)

Oversampling1000 = train_temp.loc[train_temp.money == 1000]
Oversampling1500 = train_temp.loc[train_temp.money == 1500]
Oversampling2000 = train_temp.loc[train_temp.money == 2000]
for i in range(5):
    train_temp = train_temp.append(Oversampling1000)
for j in range(8):
    train_temp = train_temp.append(Oversampling1500)
for k in range(10):
    train_temp = train_temp.append(Oversampling2000)

t_x = train_temp[predictors]
t_y = train_temp[target]    


clf = GradientBoostingClassifier(n_estimators=200,random_state=2016)


"""
other model
"""
#1
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

meta_clf = KNeighborsClassifier()
meta_clf = GradientBoostingClassifier(n_estimators=200,random_state=2016)
clf = BaggingClassifier(meta_clf, max_samples=0.5, max_features=0.5)

#2
from sklearn import cross_validation
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

clf1 = tree.DecisionTreeClassifier()
clf2 = RandomForestClassifier()
clf3 = GradientBoostingClassifier(n_estimators=200,random_state=2016)

eclf = VotingClassifier(estimators=[('df', clf1), ('rf', clf2), ('gbc', clf3)], voting='hard', weights=[1,1,2])

for clf, label in zip([clf1, clf2, clf3, eclf], ['df', 'rf', 'gbc', 'Ensemble']):
    scores = cross_validation.cross_val_score(clf,train[predictors], train[target], cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    
#3
from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1.0, probability=True,random_state=0)
clf = SVC(kernel='rbf', probability=True)


#train

clf = clf.fit(t_x,t_y)
y_pred = clf.predict(val_x)


print f1_score(y_true, y_pred,labels=[1000.0,1500.0,2000.0],average='macro')
print '1000--'+str(len(y_true[y_true==1000])) + ':prd:'+str(len(y_pred[y_pred==1000]))
print '1500--'+str(len(y_true[y_true==1500])) + ':prd:'+str(len(y_pred[y_pred==1500]))
print '2000--'+str(len(y_true[y_true==2000])) + ':prd:'+str(len(y_pred[y_pred==2000]))



"""
result !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

clf = clf.fit(train[predictors],train[target])
result = clf.predict(test[predictors])

test_result = pd.DataFrame(columns=["studentid","subsidy"])
test_result.studentid = ids
test_result.subsidy = result
test_result.subsidy = test_result.subsidy.apply(lambda x:int(x))

print 'result1000--'+str(len(test_result[test_result.subsidy==1000])) + ':741'
print 'result1500--'+str(len(test_result[test_result.subsidy==1500])) + ':465'
print 'result2000--'+str(len(test_result[test_result.subsidy==2000])) + ':354'

test_result.to_csv("output\submit2016_1217_2.csv",index=False)
