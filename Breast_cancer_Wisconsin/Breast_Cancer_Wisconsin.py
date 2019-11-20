#!/usr/bin/env python
# coding: utf-8

# # Breast cancer Wisconsin

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold


# In[2]:


# Import cancer data

dataset = pd.read_csv('C:/Users/accel/cancer.csv')
X = dataset.iloc[:, 2:32].values # 從 radius_mean 開始作為資料集
Y = dataset.iloc[:, 0].values　# 將 diagnosis 作為 label

dataset.head()


# In[3]:


print('cancer dataset dimensions:{}'.format(dataset.shape))


# In[4]:


# Define the score

def TP():
    x = 0
    for i in range(len(Pred_test)):
        if Pred_test[i] == Y_test[i]:
            if Pred_test[i] == 'B':
                x += 1
    return x

def TN():
    x = 0
    for i in range(len(Pred_test)):
        if Pred_test[i] == Y_test[i]:
            if Pred_test[i] == 'M':
                x += 1
    return x

def FP():
    x = 0
    for i in range(len(Pred_test)):
        if Pred_test[i] != Y_test[i]:
            if Pred_test[i] == 'B':
                x += 1
    return x

def FN():
    x = 0
    for i in range(len(Pred_test)):
        if Pred_test[i] != Y_test[i]:
            if Pred_test[i] == 'M':
                x += 1
    return x


# In[5]:


# 10 fold cross validation

KF = KFold(n_splits=10)

XtrainL = []
XtestL = []
YtrainL = []
YtestL = []
# split features
for X_train_index, X_test_index in KF.split(X): # X_train:X_test = 9:1, test依序從左至右
    X_train = X[X_train_index]
    X_test = X[X_test_index]
    XtrainL.append(X_train) # 儲存每次的訓練資料集
    XtestL.append(X_test) # 儲存每次的測試資料集
    
# split labels
for Y_train_index, Y_test_index in KF.split(Y):
    Y_train = Y[Y_train_index]
    Y_test = Y[Y_test_index]
    YtrainL.append(Y_train)
    YtestL.append(Y_test)


# In[6]:


predL = []
precL = []
recL =[]
f1L = []

# 儲存10次驗證的準確率
for i in range(10):
    KNN = KNeighborsClassifier()
    KNN.fit(XtrainL[i], YtrainL[i]) # train
    
    Pred_test = KNN.predict(X_test) # Predict

    Precision = TP() / (TP() + FP()) # Scoring
    Recall = TP() / (TP() + FN())
    F1 = (2 * Precision * Recall) / (Precision + Recall)
    
    # store the scores
    predL.append(Pred_test)
    precL.append(round(Precision*100, 1))
    recL.append(round(Recall*100, 1))
    f1L.append(round(F1*100, 1))
    
print('Precision of test set:{}'.format(precL))
print('Recall of test set:{}'.format(recL))
print('F1-measure of test set:{}'.format(f1L))


# In[7]:


# change label into binary
for i in range(len(YtestL[9])):
    if YtestL[9][i] == 'M':
        YtestL[9][i] = 0
    elif YtestL[9][i] == 'B':
        YtestL[9][i] = 1
        
for i in range(len(predL[9])):
    if predL[9][i] == 'M':
        predL[9][i] = 0
    elif predL[9][i] == 'B':
        predL[9][i] = 1
        
# 為符合 roc_curve 的參數規格, 將 array 轉為　list
Y_true = list(YtestL[9])
Y_pred = list(predL[9])

print(Y_true)
print(Y_pred)


# In[8]:


fpr, tpr, threshold = roc_curve(Y_true, Y_pred, pos_label=1)


# In[9]:


# roc curve & auc

roc_auc = auc(fpr,tpr) ###计算auc的值
 
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###FPR:row，TPR:column
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
 
plt.show()


# In[ ]:




