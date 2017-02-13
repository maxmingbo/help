# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 18:24:22 2016
预测
@author: 80374769
"""
import csv
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn import tree
from sklearn.cross_validation import train_test_split
tr_src = pd.read_csv(r'E:\help\train.csv')
te = pd.read_csv(r'E:\help\test.csv')

del tr_src['stu_id']




oversampling_1 = tr_src[tr_src['class']==1]
oversampling_2 = tr_src[tr_src['class']==2]
oversampling_3 = tr_src[tr_src['class']==3]
sample_0 = tr_src[tr_src['class']==0]

tr = sample_0

for i in range(len(sample_0)/len(oversampling_1)-3):
    tr = tr.append(oversampling_1)
for i in range(len(sample_0)/len(oversampling_2)-3):
    tr = tr.append(oversampling_2)    
for i in range(len(sample_0)/len(oversampling_3)-3):
    tr = tr.append(oversampling_3)
for i in range(5):
    tr = tr.append(oversampling_1)
for i in range(8):
    tr = tr.append(oversampling_2)    
for i in range(10):
    tr = tr.append(oversampling_3)
#tr = tr_src
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

clf = RandomForestClassifier()
clf = clf.fit(tr_x,tr_y)
result = clf.predict(te_x)
print len(result[result!=0])

clf_gb = GradientBoostingClassifier()
clf_gb = clf_gb.fit(tr_x_nor,tr_y)
result_gb = clf_gb.predict(te_x_nor)
print len(result_gb[result_gb!=0])

clf_tree = tree.DecisionTreeClassifier()  
clf_tree = clf_tree.fit(tr_x, tr_y)
re_tree = clf_tree.predict(te_x)
print len(re_tree[re_tree!=0])


t_x, val_x, t_y, val_y = train_test_split(tr_x, tr_y, test_size=0.2, random_state=0)


#from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.svm import SVC
#model = SVC(kernel='rbf', probability=True)

classifiers = {'RDF':RandomForestClassifier(),\
               'DF':tree.DecisionTreeClassifier(),\
                'GBC':GradientBoostingClassifier(),\
                #'LR':LogisticRegression(),\
                #'NB':MultinomialNB(alpha=0.01),\
                'KNN':KNeighborsClassifier(),\
                'SVC':SVC(kernel='rbf', probability=True)}

for name,cls in classifiers.items():
    clf = cls
    clf = clf.fit(t_x,t_y)
    pr_y = clf.predict(val_x)
    
    print name
    F = [0.0]*3
    F1 = 0.0
    for i in [1,2,3]:
        P = len(pr_y[pr_y[val_y==i]==i])*1.0/len(val_y[val_y==i])
        R = len(pr_y[pr_y[val_y==i]==i])*1.0/len(pr_y[pr_y==i])
        
#        print len(pr_y[pr_y[val_y==i]==i])
#        print len(val_y[val_y==i])
#        print len(pr_y[pr_y==i])
        
        print P,R
#        print '\n'
        F[i-1] = 2*P*R*1.0/(P+R)
        
        F1 = F1 + len(val_y[val_y==i])*F[i-1]/len(val_y)
    print F1

print abs(val_y-pr_y)

clf = SVC(kernel='rbf', probability=True)
clf = clf.fit(t_x,t_y)
pr_y = clf.predict(val_x)

print name
F = [0.0]*3
F1 = 0.0
for i in [1,2,3]:
    P = len(pr_y[pr_y[val_y==i]==i])*1.0/len(val_y[val_y==i])
    R = len(pr_y[pr_y[val_y==i]==i])*1.0/len(pr_y[pr_y==i])
    
#        print len(pr_y[pr_y[val_y==i]==i])
#        print len(val_y[val_y==i])
#        print len(pr_y[pr_y==i])
    
    print P,R
#        print '\n'
    F[i-1] = 2*P*R*1.0/(P+R)
    
    F1 = F1 + len(val_y[val_y==i])*F[i-1]/len(val_y)
print F1

"""
svm_cross_validation(train_x, train_y)
"""
def svm_cross_validation(train_x, train_y):  
    from sklearn.grid_search import GridSearchCV  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}  
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)  
    grid_search.fit(train_x, train_y)  
    best_parameters = grid_search.best_estimator_.get_params()  
    for para, val in best_parameters.items():  
        print para, val  
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)  
    model.fit(train_x, train_y)  
    return model  







re_ = clf.predict(te_x)
print len(re_[re_!=0])
"""
print out
"""
y =[]
for i in re_tree:
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
to_save = pd.DataFrame({'studentid':id,'subsidy':y})
to_save.to_csv(r're.csv',index=False)

to_save.to_csv('res_1208_2.csv',index=False,encoding='utf-8',quoting=csv.QUOTE_NONNUMERIC,line_terminator="\r")