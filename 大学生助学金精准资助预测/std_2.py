# -*-coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# train_test
train = pd.read_table(r'train\subsidy_train.txt',sep=',',header=-1)
train.columns = ['id','money']
test = pd.read_table(r'test\studentID_test.txt',sep=',',header=-1)
test.columns = ['id']
test['money'] = np.nan
train_test = pd.concat([train,test])

# score
score_train = pd.read_table('train\score_train.txt',sep=',',header=-1)
score_train.columns = ['id','college','score']
score_test = pd.read_table('test\score_test.txt',sep=',',header=-1)
score_test.columns = ['id','college','score']
score_train_test = pd.concat([score_train,score_test])

college = pd.DataFrame(score_train_test.groupby(['college'])['score'].max())
college.to_csv('input\college.csv',index=True)
college = pd.read_csv('input\college.csv')
college.columns = ['college','num']

score_train_test = pd.merge(score_train_test, college, how='left',on='college')
score_train_test['order'] = score_train_test['score']/score_train_test['num']
train_test = pd.merge(train_test,score_train_test,how='left',on='id')

# card
card_train = pd.read_table('train\card_train.txt',sep=',',header=-1)
card_train.columns = ['id','consume','where','how','time','amount','remainder']
card_test = pd.read_table('test\card_test.txt',sep=',',header=-1)
card_test.columns = ['id','consume','where','how','time','amount','remainder']
card_train_test = pd.concat([card_train,card_test])

card = pd.DataFrame(card_train_test.groupby(['id'])['consume'].count())
card['consumesum'] = card_train_test.groupby(['id'])['amount'].sum()
card['consumeavg'] = card_train_test.groupby(['id'])['amount'].mean()
card['consumemax'] = card_train_test.groupby(['id'])['amount'].max()
card['remaindersum'] = card_train_test.groupby(['id'])['remainder'].sum()
card['remainderavg'] = card_train_test.groupby(['id'])['remainder'].mean()
card['remaindermax'] = card_train_test.groupby(['id'])['remainder'].max()

card.to_csv('input\card.csv',index=True)

card = pd.read_csv('input\card.csv')
train_test = pd.merge(train_test, card, how='left',on='id')
train_test.to_csv(r'input\train_test.csv',index=False)



"""
begin
"""
train_test = pd.read_csv('input\train_test.csv')

borrow = pd.read_csv(r'input\borrow.csv')
train_test = pd.merge(train_test, borrow, how='left',on='id')


train = train_test[train_test['money'].notnull()]
test = train_test[train_test['money'].isnull()]

train = train.fillna(-1)
test = test.fillna(-1)
target = 'money'
#IDcol = 'id'
ids = test['id'].values
predictors = [x for x in train.columns if x not in [target]]
del predictors[0]

# Oversample
Oversampling1000 = train.loc[train.money == 1000]
Oversampling1500 = train.loc[train.money == 1500]
Oversampling2000 = train.loc[train.money == 2000]
for i in range(5):
    train = train.append(Oversampling1000)
for j in range(8):
    train = train.append(Oversampling1500)
for k in range(10):
    train = train.append(Oversampling2000)


#ind = list(train.index)
#import random
#random.shuffle(ind)
#train= train.iloc[ind,]
#train.index = [i for i in range(len(train))]


from sklearn.cross_validation import train_test_split
t_x, val_x, t_y, val_y = train_test_split(train[predictors], train[target], test_size=0.4, random_state=0)

t_x.index = [i for i in range(len(t_x))]
t_y.index = [i for i in range(len(t_x))]
val_x.index = [i for i in range(len(val_x))]
val_y.index = [i for i in range(len(val_y))]
               
"""
print F1
"""
clf = GradientBoostingClassifier(n_estimators=200,random_state=2016)  
#0.331335280627(0.4)   0.390588318597(加入borrow数据)
#0.41232047069(0.2)

#等分较高
#1000--1077:741
#1500--597:465
#2000--428:354
#+borrow
#1000--1039:741
#1500--592:465
#2000--433:354
clf = GradientBoostingClassifier()  

clf = RandomForestClassifier()
#0.56062581634(0.4)
#0.560870991495(0.2)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
#0.54130913826(0.2)
#0.539993025782(0.4)
#1000--795:741
#1500--456:465
#2000--418:354

from sklearn.svm import SVC
clf = SVC(kernel='rbf', probability=True)
#0.571624713959(0.2)
#0.57538613173 (0.4)

clf = clf.fit(t_x,t_y)
pr_y = clf.predict(val_x)

F = [0.0]*3
n=[0,1,2]
mone = [1000,1500,2000]
F1 = 0.0
for j,m in zip(n,mone):
    b = len(val_y[val_y==m])
    a = len(pr_y[pr_y==m])
    list_m = []
    c = 0
    for index,key in enumerate(val_y):
        if key == m:
            list_m.append(index)
    for k in pr_y[list_m]:
        if k == m:
            c += 1    
    P = c*1.0/b
    R = c*1.0/a 
#    print P,R
    F[j-1] = 2*P*R*1.0/(P+R)   
    F1 = F1 + len(val_y[val_y==m])*F[j-1]/len(val_y)
print F1


# model
clf = GradientBoostingClassifier(n_estimators=200,random_state=2016)
# clf = RandomForestClassifier(n_estimators=500,random_state=2016)


clf = clf.fit(train[predictors],train[target])

result = clf.predict(test[predictors])

# Save results
test_result = pd.DataFrame(columns=["studentid","subsidy"])
test_result.studentid = ids
test_result.subsidy = result
test_result.subsidy = test_result.subsidy.apply(lambda x:int(x))


print '1000--'+str(len(test_result[test_result.subsidy==1000])) + ':741'
print '1500--'+str(len(test_result[test_result.subsidy==1500])) + ':465'
print '2000--'+str(len(test_result[test_result.subsidy==2000])) + ':354'


test_result.to_csv("output\submit2016_1213_2_GBC.csv",index=False)


# '''