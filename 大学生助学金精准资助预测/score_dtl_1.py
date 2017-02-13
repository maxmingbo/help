# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 17:38:16 2016
成绩处理
@author: 80374769
"""

import csv
import numpy as np
import pandas as pd

score_train = pd.read_csv(r'E:\help\train\score_train.txt')
score_train.columns = ('stu_id','dep','top')

#train_dep_tot = score_train.groupby('dep').count()
#train_dep_tot.columns = ('tot_num','tot_num1')
train_dep_tot = [0]*19
for i in range(1,20):
    train_dep_tot[i-1] = int(score_train[score_train['dep']==i].top.max())

top_ =[]
for i in score_train.index:    
    top_.append(score_train.at[i,'top']*100.0/train_dep_tot[score_train.at[i,'dep']-1])

score_train['topN'] = pd.Series(np.array(top_),index=score_train.index)

#score_train.to_csv(r'E:\help\score_train.csv',index = False)







score_test = pd.read_csv(r'E:\help\test\score_test.txt')
score_test.columns = ('stu_id','dep','top')

#test_dep_tot = score_test.groupby('dep').count()
#test_dep_tot.columns = ('tot_num','tot_num1')
test_dep_tot = [0]*19

for i in range(1,20):
    test_dep_tot[i-1] = int(score_test[score_test['dep']==i].top.max())
    
top_ =[]
for i in score_test.index:    
    top_.append(score_test.at[i,'top']*100.0/test_dep_tot[score_test.at[i,'dep']-1])

score_test['topN'] = pd.Series(np.array(top_),index=score_test.index)
#score_test.to_csv(r'E:\help\score_test.csv',index = False)

score = pd.concat([score_train,score_test])

score.to_csv(r'E:\help\score.csv',index=False)


