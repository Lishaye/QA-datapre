from numpy import *
import pandas as pd
import time
import numpy as np
import os
import csv
#读取csv文件，获取要计算feature的modle
data1=pd.read_csv("/public/home/yels/project/CASP/CASP10/casp10_label/global_s1.csv")
df= pd.DataFrame(data1) 
print(df.shape[0])
L=[]
L.append(int(0))
for j in range(0,df.shape[0]-1):
    d1=df.loc[j].values[0]
    t1=d1.split('-')[0]
    d2=df.loc[j+1].values[0]
    t2=d2.split('-')[0]
    if t1!=t2:
       L.append(j+1)
length=len(L)
print(length)
L2=[]
for i in range(0,length):
    a=L[i]
    if i <length-1:
       b=L[i+1]
    else:
       b=df.shape[0]
    d1=df.loc[a].values[0]
    d2=float(df.loc[a].values[1])
    target=d1.split("-")[0] 
    path1='/public/home/yels/project/CASP/CASP10/casp10_feature_s1/feature1_label_g/'+target+'.npy'
    s1=np.load(path1)
    print(s1.shape)
    path2='/public/home/yels/project/CASP/CASP10/casp10_feature_s2/feature11_label_g/'+target+'.npy'
    s2=np.load(path2)
    print(s2.shape)
    L2.append(s2)
    s=np.vstack((s1,s2))  
    print(s.shape)
    L2.append(s)
print(len(L2))
print(type(L2))
path3='/public/home/yels/project/CASP/CASP10/casp10_feature/global31/global.npy'
np.save(path3,L2)