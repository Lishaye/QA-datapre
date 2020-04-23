# -*- coding: utf-8 -*-
import os 
import subprocess
import numpy as np
import pandas as pd
import math 
import csv
from clu_function import *
def file_name(file_dir):  
  L=[]  
  for root, dirs, files in os.walk(file_dir): 
    for file in files: 
        L.append(os.path.join(root, file))
        #L.append(file)
  return L
data1=pd.read_csv("/public/home/yels/project/CASP/CASP10/casp10_label/global_s2.csv")
df= pd.DataFrame(data1) 
L=[]
L.append(int(0))
for j in range(0,df.shape[0]-1):
    d1=df.loc[j].values[0]
    t1=d1.split('-')[0]
    #print(t1)
    d2=df.loc[j+1].values[0]
    t2=d2.split('-')[0]
    #print(t2)
    if t1!=t2:
       L.append(j+1)
print(L)
length=len(L)
print(length)
for i in range(0,length):
    a=L[i]
    if i <length-1:
       b=L[i+1]
    else:
       b=df.shape[0]
    d1=df.loc[a].values[0]
    d2=float(df.loc[a].values[1])
    target=d1.split("-")[0] 
    path='/public/home/yels/project/CASP/CASP10/casp10_feature_s2/mtx/'+target+'.npy'
    C=cluster(path)
    print(C.shape)
    path1='/public/home/yels/project/CASP/CASP10/casp10_feature_s2/cluster/'+target+'.npy'
    np.save(path1,C)
#####123#####

