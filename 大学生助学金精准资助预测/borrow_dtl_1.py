# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 19:11:19 2016
借书数量处理

@author: 80374769
"""

import csv
import numpy as np
import pandas as pd
#import datetime

bor_src = open(r'E:\help\train\borrow_train.txt','r')
reader = csv.reader(bor_src)
#print reader

c=0
stu_id = []
day = []
for line in reader:
    #print line[:2]
    stu_id.append(line[0])
    d = sum(np.array([int(i) for i in line[1].split('/')])*np.array([10000,100,1]))
#    if d-20140901>=0:
#        print line[1]
    day.append(d)

bor_src.close()


bor_data_1 = pd.DataFrame({'stu_id':np.array(stu_id),\
                         'day':np.array(day),\
                         'times':np.array([1]*len(day))})


bor_src_t = open(r'E:\help\test\borrow_test.txt','r')
reader_t = csv.reader(bor_src_t)
#print reader


stu_id = []
day = []
for line in reader_t:
    #print line[:2]
    stu_id.append(line[0])
    d = sum(np.array([int(i) for i in line[1].split('/')])*np.array([10000,100,1]))

    day.append(d)
bor_src_t.close()


bor_data_2 = pd.DataFrame({'stu_id':np.array(stu_id),\
                         'day':np.array(day),\
                         'times':np.array([1]*len(day))})

bor_data = pd.concat([bor_data_1, bor_data_2])

bor = bor_data.groupby('stu_id').sum()

bor.to_csv(r'E:\help\borrow.csv')
#
#
#bor_train = bor_data[bor_data['day']<0].groupby('stu_id').sum()
#bor_test = bor_data[bor_data['day']>=0].groupby('stu_id').sum()
#
#
#
#bor_train.to_csv(r'E:\help\borrow_train.csv')
#bor_test.to_csv(r'E:\help\borrow_test.csv')








