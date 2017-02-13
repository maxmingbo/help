# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 18:40:48 2016
助学金标签
@author: 80374769
"""

import pandas as pd
import numpy as np

subsidy = pd.read_csv(r'E:\help\train\subsidy_train.txt')

subsidy.columns = ('stu_id','help_money')

cls = []

for i in subsidy.index:
    if subsidy.at[i,'help_money'] == 0:
        cls.append(0)
    elif subsidy.at[i,'help_money'] == 1000:
        cls.append(1)
    elif subsidy.at[i,'help_money'] == 1500:
        cls.append(2)
    elif subsidy.at[i,'help_money'] == 2000:
        cls.append(3)
    else:
        print subsidy.at[i,'help_money']
        
subsidy['class'] = pd.Series(np.array(cls),index=subsidy.index)

subsidy.to_csv(r'E:\help\help_class.csv',index=False)


        