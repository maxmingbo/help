# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 21:34:54 2016

@author: zmb
"""

import pandas as pd
import numpy as np

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression 
import csv
import random


pos=pd.read_csv(r'E:\tianchi\o2o\postive.csv')
neg=pd.read_csv(r'E:\tianchi\o2o\neg.csv')


data = pd.concat([pos,neg],ignore_index=True)  


data =pd.read_csv(r'E:\tianchi\o2o\train_of_shuffle_all_01.csv')

'''
加入m_rate数据

'''

M_rate = pd.read_csv(r'E:\tianchi\o2o\M_rate.csv',header=0,sep=',')
M = {}
for i in M_rate.index:
    M[M_rate.at[i,'Merchant_id']] = M_rate.at[i,'use_rate']
    
m_rate=[]   
j = 0
for i in data.Merchant_id.values:
    if i in M.keys():
        m_rate.append(M[i])
    else:
        m_rate.append(0)
        j += 1
        print(i)
print(j)

data['m_rate']=pd.Series(np.array(m_rate),index=data.index) 
'''
结束加入m_rate数据
'''


train = data[['Distance','base','dis','m_rate','use_cop']]

#inde = train_data.index
#
#inde = list(inde)
#random.shuffle(inde)
#train = train_data.ix[inde]
#train.to_csv('train_of_shuffle_all_01.csv')



#X = train[['Distance','weekday','base','m_rate','dis']]
X = train[['Distance','base','m_rate','dis']]
y = train['use_cop']

test_src = pd.read_csv(r'E:\tianchi\o2o\test_data_dtl.csv',header=0,sep=',')
#test = test_src[['Distance','weekday','base','m_rate','dis']]
test = test_src[['Distance','base','m_rate','dis']]


X_test = pd.concat([X,test],ignore_index=True)


from sklearn import preprocessing 
X_test = preprocessing.scale(X_test)

#归一化0-1：min_max_scaler = preprocessing.MinMaxScaler()
#归一化X_train_minmax = min_max_scaler.fit_transform(X_train)
#min_max_scaler = preprocessing.MinMaxScaler()

#X_test = min_max_scaler.fit_transform(X_test)

X = X_test[0:len(X)]
test = X_test[len(X):]

#一起归一化，后再拆分


tr_x, te_x, tr_y, te_y = cross_validation.train_test_split(X, y, test_size=0.1, random_state=0)


from sklearn.externals import joblib

#lr0 = LogisticRegression(penalty='l1',class_weight='auto')

lr0 = LogisticRegression(penalty='l1')
lr0.fit(tr_x, tr_y)

#调用 lr0 = joblib.load('lr0.model') # er = 31.5
#调用  er = 24.22

pr = lr0.predict(te_x)
er = (abs(pr-te_y)).sum()
pro = lr0.predict_proba(te_x)[:,1]

from sklearn.metrics import roc_auc_score


print('auc is : %f'% roc_auc_score(te_y,pro))
print('error number rate is :%f '%(er*1.0/len(te_y)*100))

print('positive in all rate is :%f'%(pr.sum()*1.0/len(te_y)*100))

#error number rate is :22.547447 
#positive in all rate is :25.630168

from sklearn.svm import SVC
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

C = 1.0
#kernel = 1.0 * RBF([1.0, 1.0])  # for GPC

classifiers = {'L1 logistic': LogisticRegression(C=C, penalty='l1'),
               'L2 logistic (OvR)': LogisticRegression(C=C, penalty='l2'),
#               'Linear SVC': SVC(kernel='linear', C=C, probability=True,
#                                 random_state=0),
               'L2 logistic (Multinomial)': LogisticRegression(
                C=C, solver='lbfgs', multi_class='multinomial'),
               #'GPC': GaussianProcessClassifier(kernel),
                'rf': RandomForestClassifier(random_state=123),
                'GNB': GaussianNB()              
               }
               
for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(tr_x, tr_y)

    y_pred = classifier.predict(te_x)
    
    class_pro = classifier.predict_proba(te_x)[:,1]
    
    print('auc is : %f'% roc_auc_score(te_y,class_pro))
    #classif_err = ((y_pred-te_y)**2).sum()*1.0/len(te_y) * 100
    
    classif_err = abs(y_pred-te_y).sum()*1.0/len(te_y) * 100
    print("classif_rate for %s : %f " % (name, classif_err))
    
    print(y_pred.sum()*100.0/len(y_pred))    



#score = [1.1,0.2,3.4,5.1,2.1,0.4,2.4]
#label = [0,0,1,0,1,0,0]
#auc = roc_auc_score(label,score)
#print 'AUC:',auc

#classif_rate for Linear SVC : 25.378162 
#classif_rate for rf : 22.236512 
#classif_rate for GNB : 24.404434 
#classif_rate for L2 logistic (OvR) : 24.239084 
#classif_rate for L1 logistic : 24.226836 
#classif_rate for L2 logistic (Multinomial) : 24.232960            

#test_src = pd.read_csv('test_data_dtl.csv',header=0,sep=',')
#
##test = test_src[['Distance','weekday','base','m_rate','dis']]
#
#test = test_src[['Distance','base','m_rate','dis']]

#test= preprocessing.scale(test)





lr = LogisticRegression(penalty='l1',class_weight='auto')

lr = RandomForestClassifier(random_state=123)


lr.fit(X, y)
joblib.dump(lr,r'.\model\all_01_lr_auto.model')



pr_y = lr0.predict(test)
pro_y = lr0.predict_proba(test)[:,1]

print(pr_y.sum()*100.0/len(pr_y))

#38.18 scaler with X 
#41.47431779582714



an = test_src[['User_id','Coupon_id','Date_received']]

an['Probability'] = pd.Series(np.array(pro_y),index=test_src.index)  

an.to_csv(r'E:\tianchi\o2o\ans\all_01_scaler_lr_auto.csv',index=False)








