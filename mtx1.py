# -*- coding: utf-8 -*-
import os 
import subprocess
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
import math 
import csv
def file_name(file_dir):  
  L=[]  
  for root, dirs, files in os.walk(file_dir): 
    for file in files: 
        L.append(os.path.join(root, file))
        #L.append(file)
  return L
data1=pd.read_csv("/public/home/yels/project/CASP/CASP11/casp11_label/global_s2.csv")
#data1=pd.read_excel("/public/home/yels/project/Testmodel/test1.xlsx")
df= pd.DataFrame(data1) 
L=[]
L.append(int(0))
for j in range(0,df.shape[0]-1):
    d1=df.loc[j].values[0]
    t1=d1.split('-')[0]
    d2=df.loc[j+1].values[0]
    t2=d2.split('-')[0]
    if t1!=t2:
       L.append(j+1)
print(L)
length=len(L)
print(length)
for i in range(0,length):
    L2=[]
    a=L[i]
    if i <length-1:
       b=L[i+1]
    else:
       b=df.shape[0]
    print(a,b)
    l=[]
    for i in range(a,b-1):
        model=df.loc[i][0]
        target=model.split('-')[0]
        if len(target)>=7:
           name1=model[8:]
        else:
           name1=model[6:]
        path='/public/home/yels/project/CASP/CASP11/casp11_stage2/'+target
        M1=path+'/'+name1
        #print(M1)
        for j in range (i+1,b):
            model1=df.loc[j][0]
            if len(target)>=7:
               name2=model1[8:]
            else:
               name2=model1[6:]
            path1='/public/home/yels/project/CASP/CASP11/casp11_stage2/'+target
            M2=path1+'/'+name2
            #print(M2)
            args='./a.out'+' '+M1+' '+M2
            #print(args)
            s=subprocess.Popen(args,bufsize=0,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
            list=[]
            lines=[]
            while True:
                  nextline=s.stdout.readline()
                  if nextline=="" :
                     break
                  else:
                     list.append(nextline)
            arry=np.array(list)
            #print(arry)
            lines=arry.astype(dtype=np.float)
            lines[0]=round(lines[0],3)
            print(lines[0])
            l.append(lines[0])
    mtx=np.array(l)  
    print(mtx.shape)    
    path='/public/home/yels/project/CASP/CASP11/casp11_feature_s2/mtx/'+target+'.npy'
    np.save(path,mtx)
         


