# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 20:00:16 2019

@author: HP
"""
from numpy import *
import pandas as pd
import time
import numpy as np
import os
import csv
# data1=np.load('/public/home/yels/project/CASP/CASP10/casp10_feature/global36/globals1.npy',allow_pickle=True)
# data2=np.load('/public/home/yels/project/CASP/CASP10/casp10_feature/global36/globals2.npy',allow_pickle=True)
# data3=np.load('/public/home/yels/project/CASP/CASP11/casp11_feature/global36/globals1.npy',allow_pickle=True)
# data4=np.load('/public/home/yels/project/CASP/CASP11/casp11_feature/global36/globals2.npy',allow_pickle=True)
# data5=np.load('/public/home/yels/project/CASP/CASP12/casp12_feature/global36/globals1.npy',allow_pickle=True)
# data6=np.load('/public/home/yels/project/CASP/CASP12/casp12_feature/global36/globals2.npy',allow_pickle=True)
# data7=np.load('/public/home/yels/project/CASP/CASP13/casp13_feature/global36/globals1.npy',allow_pickle=True)
# data8=np.load('/public/home/yels/project/CASP/CASP13/casp13_feature/global36/globals2.npy',allow_pickle=True)
# a1=data1.tolist()
# a2=data2.tolist()
# a3=data3.tolist()
# a4=data4.tolist()
# a5=data5.tolist()
# a6=data6.tolist()
# a7=data7.tolist()
# a8=data8.tolist()

# print(len(a1),type(a1))
# print(len(a2),type(a2))
# print(len(a3),type(a3))
# print(len(a4),type(a4))
# print(len(a5),type(a5))
# print(len(a6),type(a6))
# print(len(a7),type(a7))
# print(len(a7),type(a8))
# l1=[]
# l2=[]
# l3=[]
# l4=[]
# for i in range(len(a1)):
    # b=np.empty((0,37))
    # b=np.vstack((b,data1[i]))
    # b=np.vstack((b,data2[i]))
    # l1.append(b)
# for i in range(len(a3)):
    # b=np.empty((0,37))
    # b=np.vstack((b,data3[i]))
    # b=np.vstack((b,data4[i]))
    # l2.append(b)  
# for i in range(len(a5)):
    # b=np.empty((0,37))
    # b=np.vstack((b,data5[i]))
    # b=np.vstack((b,data6[i]))
    # l3.append(b)
# for i in range(0,39):
    # b=np.empty((0,37))
    # b=np.vstack((b,data7[i]))
    # b=np.vstack((b,data8[i]))
    # l4.append(b)    
    
    
    
    
# print(len(l1))
# print(len(l2))
# print(len(l3))
# l1[len(l1):len(l1)]=l2
# l1[len(l1):len(l1)]=l3
# l1[len(l1):len(l1)]=l4
# print(len(l1))
# np.save('/public/home/yels/project/CASP/Global/targets1.npy',l
# b=np.empty((0,37))
# for i in range (len(l1)):
   # b=np.vstack((b,l1[i]))
# print(b.shape)
# np.save("/public/home/yels/project/CASP/Global/stage1/global36.npy",b)



# data1=np.load('/public/home/yels/project/CASP/CASP10/casp10_feature/globals2/global48.npy',allow_pickle=True)
# data2=np.load('/public/home/yels/project/CASP/CASP11/casp11_feature/globals2/global48.npy',allow_pickle=True)
# data3=np.load('/public/home/yels/project/CASP/CASP12/casp12_feature/globals2/global48.npy',allow_pickle=True)
# a1=data1.tolist()
# a2=data2.tolist()
# a3=data3.tolist()
# a1[len(a1):len(a1)]=a2
# a1[len(a1):len(a1)]=a3
# print(len(a1))
# np.save('/public/home/yels/project/CASP/Global/targets2.npy',a1)













# data1=np.load('/public/home/yels/project/CASP/CASP10/casp10_feature_s1/Local/feature/local2.npy',allow_pickle=True)
# data2=np.load('/public/home/yels/project/CASP/CASP10/casp10_feature_s2/Local/feature/local2.npy',allow_pickle=True)
# data3=np.load('/public/home/yels/project/CASP/CASP11/casp11_feature_s1/Local/feature/local2.npy',allow_pickle=True)
# data4=np.load('/public/home/yels/project/CASP/CASP11/casp11_feature_s2/Local/feature/local2.npy',allow_pickle=True)
# data5=np.load('/public/home/yels/project/CASP/CASP12/casp12_feature_s1/Local/feature/local2.npy',allow_pickle=True)
# data6=np.load('/public/home/yels/project/CASP/CASP12/casp12_feature_s2/Local/feature/local2.npy',allow_pickle=True)
# a1=data1.tolist()
# a2=data2.tolist()
# a3=data3.tolist()
# a4=data4.tolist()
# a5=data5.tolist()
# a6=data6.tolist()
# print(len(a1),type(a1))
# print(len(a2),type(a2))
# print(len(a3),type(a3))
# print(len(a4),type(a4))
# print(len(a5),type(a5))
# print(len(a6),type(a6))
# a1[len(a1):len(a1)]=a2
# a1[len(a1):len(a1)]=a3
# a1[len(a1):len(a1)]=a4
# a1[len(a1):len(a1)]=a5
# a1[len(a1):len(a1)]=a6
# print(len(a1),type(a1))
# List=[]
# for i in range(len(a1)):
    # b=np.empty((0,561))
    # print(a1[i].shape)
    # for j in range(a1[i].shape[0]):
        # if a1[i][j,560]!=-1:
           # b=np.vstack((b,a1[i][j,:]))
    # List.append(b)
# print(len(List))
# np.save('/public/home/yels/project/CASP/Local/train/traindup.npy',List)   



# data1=np.load('/public/home/yels/project/CASP/CASP13/casp13_feature_s1/Local/feature/local2.npy',allow_pickle=True)
# # data2=np.load('/public/home/yels/project/CASP/CASP13/casp13_feature_s2/Local/feature/local.npy',allow_pickle=True)
# print(len(data1))
# list1=[]
# for i in range(len(data1)): #59
    # list2=[]
    # for j in range(len(data1[i])):#20
        # b=np.empty((0,561))
        # for k in range(data1[i][j].shape[0]): #L
            # if data1[i][j][k,560]!=-1:
               # b=np.vstack((b,data1[i][j][k,:]))
        # list2.append(b)
    # print(len(list2))
    # list1.append(list2)        
# print(len(list1))
# np.save('/public/home/yels/project/CASP/Local/test/locals1dup.npy',list1)

data=np.load('/public/home/yels/project/CASP/Local/train/traindup.npy',allow_pickle=True)
a1=data.tolist()
print(len(data))
data1=np.load('/public/home/yels/project/CASP/Local/test/locals1dup.npy',allow_pickle=True)
data2=np.load('/public/home/yels/project/CASP/Local/test/locals2dup.npy',allow_pickle=True)
data1=data1.tolist()
data2=data2.tolist()
print(len(data1))
print(len(data2))
l1=[]
l2=[]
for i in range(0,39):
    for j in range(len(data1[i])):
         l1.append(data1[i][j])
for i in range(0,39):
    for j in range(len(data2[i])):
         l2.append(data2[i][j])
print(len(l1))
print(len(l2))
a1[len(a1):len(a1)]=l1
a1[len(a1):len(a1)]=l2
print(len(a1))
data1=data1[39:]
data2=data2[39:]
print(len(data1))
print(len(data2))
np.save('/public/home/yels/project/CASP/Local/train/train.npy',a1)
np.save('/public/home/yels/project/CASP/Local/test/locals1.npy',data1)
np.save('/public/home/yels/project/CASP/Local/test/locals2.npy',data2)
























# data1=np.load('/public/home/yels/project/CASP/CASP10/casp10_feature/globals2/global48.npy',allow_pickle=True)
# data2=np.load('/public/home/yels/project/CASP/CASP11/casp11_feature/globals2/global48.npy',allow_pickle=True)
# data3=np.load('/public/home/yels/project/CASP/CASP12/casp12_feature/globals2/global48.npy',allow_pickle=True)
# a1=data1.tolist()
# a2=data2.tolist()
# a3=data3.tolist()
# a1[len(a1):len(a1)]=a2
# a1[len(a1):len(a1)]=a3
# print(len(a1))
# # print(len(a1))
# # b=np.empty((0,48))
# # for i in range(len(a1)):
    # # b=np.vstack((b,a1[i]))
# # print(b.shape)
# # np.save("/public/home/yels/project/CASP/Global/stage2/global48s2.npy",b)
# # data=np.load('/public/home/yels/project/CASP/Global/stage2/global48s2.npy')
# # print(data.shape)
# # # data1=np.load('/public/home/yels/project/CASP/CASP13/casp13_feature/global36/globals1.npy')
# data4=np.load('/public/home/yels/project/CASP/CASP13/casp13_feature/globals2/global48.npy',allow_pickle=True)
# a4=data4.tolist()
# a4=a4[0:39]
# a1[len(a1):len(a1)]=a4
# print(len(a1))
# np.save('/public/home/yels/project/CASP/Global/stage2/globaltarget_s2',a1)
# print(len(data1))
# # print(len(data2))
# b=np.empty((0,48))
# for i in range(0,39):
    # b=np.vstack((b,data1[i]))
# print(b.shape)
# data=np.vstack((data,b))
# print(data.shape)
# np.save('/public/home/yels/project/CASP/Global/stage2/global48_casp10_13.npy',data)
# data1=data1[39:]
# # data2=data2[39:]
# print(len(data1))
# # print(len(data2))
# np.save('/public/home/yels/project/CASP/Global/stage2/global48tests2.npy',data1)
# np.save('/public/home/yels/project/CASP/Global/stage1/global36tests2.npy',data2)






