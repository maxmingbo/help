# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 18:56:08 2016
一卡通数据处理
@author: 80374769
"""
import sys
print sys.getdefaultencoding()


import pandas as pd
import numpy as np

card_data = pd.read_csv(r'E:\help\train\card_train.txt')
card_2 = pd.read_csv(r'E:\help\test\card_test.txt')

card_data.columns=('stu_id','consum_cls','addr','con_pat','date','money','left_money')
card_2.columns=('stu_id','consum_cls','addr','con_pat','date','money','left_money')

card_data_1 = card_data[card_data.con_pat =='校车']
card_data_2 = card_2[card_2.con_pat =='校车']

card = pd.concat([card_data_1,card_data_2])
del card['consum_cls']
del card['addr']
del card['con_pat']

#card_train = card[card.date< '2014/09/01 00:00:00']
#card_test = card[card.date>= '2014/09/01 00:00:00']




card_m_sum = card.groupby('stu_id').money.sum()
card_m_count = card.groupby('stu_id').money.count()
card_avg = card_m_sum.values/card_m_count.values

card_to_save = pd.DataFrame({'stu_id':card_m_sum.index,\
                                   'ling_sum':card_m_sum,\
                                   'ling_count':card_m_count,\
                                   'avg_ling':card_avg
                                   })

card_to_save.to_csv(r'E:\help\card_ling.csv',index=False)


"""
max
"""

card = pd.concat([card_data,card_2])
card_left_avg = card.groupby('stu_id').left_money.max()

card_to_save = pd.DataFrame({'stu_id':card_left_avg.index,\
                                   'left_money_max':card_left_avg
                                   })
card_to_save.to_csv(r'E:\help\card_left_max.csv',index=False)

"""
mean
"""

card_left_avg = card.groupby('stu_id').left_money.mean()

card_to_save = pd.DataFrame({'stu_id':card_left_avg.index,\
                                   'left_money_avg':card_left_avg
                                   })
card_to_save.to_csv(r'E:\help\card_left_avg.csv',index=False)


"""
test数据处理
"""
#card_test_m_sum = card_test.groupby('stu_id').money.sum()
#card_test_m_count = card_test.groupby('stu_id').money.count()
#card_test_avg = card_test_m_sum.values/card_test_m_count.values
#
#card_test_to_save = pd.DataFrame({'stu_id':card_test_m_sum.index,\
#                                   'mon_sum':card_test_m_sum,\
#                                   'mon_count':card_test_m_count,\
#                                   'avg':card_test_avg
#                                   })
#
#card_test_to_save.to_csv(r'E:\help\test_card_c.csv',index=False)
#
#
pat = set(card_data.con_pat.values)

for x in pat:
    print x