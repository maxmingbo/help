# -*- coding: utf-8 -*-
"""
Created on Mon Dec 05 22:55:16 2016
train data merge
test data merge
@author: zmb
"""

import numpy as np


#book = pd.read_csv(r'E:\help\borrow.csv')[['stu_id','times']]
#
#book_avg = book.times.mean()
#
#card_s = pd.read_csv(r'E:\help\card_s.csv')[['stu_id','avg_s']]
#
#card_c = pd.read_csv(r'E:\help\card_c.csv')[['stu_id','avg_c','mon_count_c']]
#score = pd.read_csv(r'E:\help\score.csv')[['stu_id','topN']]
##score.columns=('stu_id','top')
#
#tr_id_cls=pd.read_csv(r'E:\help\help_class.csv')[['stu_id','class']]
#
#
#train_src = pd.merge(tr_id_cls,card_s,how='left',on='stu_id')
#train_src = train_src[~train_src.avg_s.isnull()]
#                      
#train_src = pd.merge(train_src,card_c,how='left',on='stu_id')
#train_src = train_src[~train_src.avg_c.isnull()]
#                      
#train_src = pd.merge(train_src,score,how='left',on='stu_id')
#train_src = train_src.fillna(method='bfill')
#
#train_src = pd.merge(train_src,book,how='left',on='stu_id')
#train_src = train_src.fillna(0)

#train_src.to_csv(r'E:\help\tr_nan.csv')
#
#
#
#                      
##train_src = train_src[~train_src.topN.isnull()]
#
#train_src.index = [i for i in range(len(train_src))]
#train_src = train_src.ix[0:7500]
#
#train_src = train_src.fillna(0)


#
#train_src.to_csv(r'E:\help\tr_dropnan.csv',index= False)

#tr_dtl_1 = train_src.dropna(how='any')







#te_id=pd.read_csv(r'E:\help\studentID_test.txt',header=-1)
#te_id.columns = ('stu_id',)
#
#te_src = pd.merge(te_id,card_s,how='left',on='stu_id')
#te_src = te_src.fillna(te_src.avg_s.mean())
#
#te_src = pd.merge(te_src,card_c,how='left',on='stu_id')
#te_src = te_src.fillna(te_src.mean()['avg_c':'mon_count_c'])
#
#te_src = pd.merge(te_src,score,how='left',on='stu_id')
#te_src = te_src.fillna(method='bfill')
#
#te_src = pd.merge(te_src,book,how='left',on='stu_id')
#te_src = te_src.fillna(0)

#te_dtl_1 = te_src.dropna(how='any')

#te_src.to_csv(r'E:\help\te.csv',index=False)

import pandas as pd

tr = pd.read_csv(r'E:\help\tr_dropnan.csv')
te = pd.read_csv(r'E:\help\te.csv')

card_left = pd.read_csv(r'E:\help\card_left_avg.csv')
card_ling = pd.read_csv(r'E:\help\card_ling.csv')
card_water = pd.read_csv(r'E:\help\card_water.csv')
card_xi = pd.read_csv(r'E:\help\card_xi.csv')

data_add_list = [card_left,card_ling,card_water,card_xi]
for data in data_add_list:
    tr = pd.merge(tr,data,how = 'left',on='stu_id')    
    te = pd.merge(te,data,how = 'left',on='stu_id')
    
    
tr = tr.fillna(0)
te = te.fillna(te.mean()['left_money_avg':'left_money_avg'])
te = te.fillna(0)

tr.to_csv(r'E:\help\train.csv',index=False)
te.to_csv(r'E:\help\test.csv',index=False)
#tr = pd.merge(tr,card_left,how='left',on='stu_id')
##tr = tr[~tr.left_money_avg.isnull()]
#
#tr = pd.merge(tr,card_ling,how='left',on='stu_id')
#tr = tr.fillna(0)  #['avg_ling':'ling_sum'])
#
#tr = pd.merge(tr,card_water,how='left',on='stu_id')
#tr = tr.fillna(0)
#
#tr = pd.merge(tr,card_xi,how='left',on='stu_id')
#tr = tr.fillna(0)

import pandas as pd

tr = pd.read_csv(r'E:\help\train.csv')
te = pd.read_csv(r'E:\help\test.csv')

card_left_max = pd.read_csv(r'E:\help\card_left_max.csv')

tr = pd.merge(tr,card_left_max,how = 'left',on='stu_id')    
te = pd.merge(te,card_left_max,how = 'left',on='stu_id')
    
    
tr = tr.fillna(0)
te = te.fillna(te.mean()['left_money_avg':'left_money_avg'])
te = te.fillna(0)

tr.to_csv(r'E:\help\train.csv',index=False)
te.to_csv(r'E:\help\test.csv',index=False)




