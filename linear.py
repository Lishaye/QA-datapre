# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 20:00:16 2019

@author: HP
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
def corr(y_true, y_pred):
    y_true=y_true.reshape(y_true.shape[0],1)
    y_pred=y_pred.reshape(y_pred.shape[0],1)
    C=np.hstack((y_true,y_pred))
    C=C.T
    data=np.corrcoef(C)
    data=np.around(data, decimals=4)
    data=data.tolist()
    corr=data[0][1]
    return corr
def best_different(y_true, y_pred):
    p=y_pred.tolist()
    t=y_true.tolist()
    index1=p.index(max(p))
    index2=t.index(max(t))
    diff=abs(t[index1]-t[index2])*100
    return diff
def target_loss(y_true, y_pred):
    x=y_true*100
    y=y_pred*100
    z=np.mean(abs(x-y))
    return z
model=LinearRegression()
# train=np.load("/public/home/yels/project/CASP/Global/stage1/global36.npy")
# tests=np.load("/public/home/yels/project/test/feature/feature36.npy",allow_pickle=True)
#train=np.load("/public/home/yels/project/CASP/Global/stage1/global29.npy")
train=np.load("/public/home/yels/project/CASP/Global/stage2/global48s2.npy")
#tests=np.load("/public/home/yels/project/CASP/CASP13/casp13_feature/global29/globals1.npy")
#tests=np.load("/public/home/yels/project/CASP/CASP13/casp13_feature/global29/globals2.npy",allow_pickle=True)
#tests=np.load("/public/home/yels/project/test/feature/feature.npy",allow_pickle=True)
tests=np.load("/public/home/yels/project/CASP/CASP13/casp13_feature/globals2/global48.npy",allow_pickle=True)
x_train=train[:,:-1]
y_train=train[:,-1]
# x_test=tests[:,:,:-1]
# y_test=tests[:,:,-1]
print("x_train.shape:",x_train.shape)
print("y_train.shape:",y_train.shape)
print(len(tests))
# print("x_test.shape:",x_test.shape)
# print("y_test.shape:",y_test.shape)
model.fit(x_train,y_train)
print(model.coef_)               #输出多元线性回归的各项系数
print(model.intercept_)          #输出多元线性回归的常数项的值
corr_value = np.zeros(len(tests))
best_diff = np.zeros(len(tests))
tar_loss = np.zeros(len(tests))
for i in range(len(tests)):
    # x=x_test[i]
    # y_true=y_test[i]
    x=tests[i][:,:-1]
    y_true=tests[i][:,-1]
    y_pred=model.predict(x)
    corr_value[i]=corr(y_true, y_pred)
    best_diff[i]=best_different(y_true, y_pred)
    tar_loss[i]=target_loss(y_true, y_pred)
print("target_loss：",tar_loss)
print("\npearson_corr:",corr_value)
print("\nbest_diff: ", best_diff)
print("\nmean target_loss：",np.mean(tar_loss))
print("\nmean pearson_corr:",np.mean(corr_value))
print("\nmean best_diff: ", np.mean(best_diff))


