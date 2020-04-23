# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 20:00:16 2019

@author: HP
"""
import time
import numpy as np
import os
import pandas as pd
import csv
def import_DLS2FSVM1(filename, delimiter='\t', delimiter2=' ',comment='>',skiprows=0, start=0, end = 0,target_col = 1, dtype=np.float32):
    # Open a file
    file = open(filename, "r")
    #print "Name of the file: ", file.name
    if skiprows !=0:
       dataset = file.read().splitlines()[skiprows:]
    if skiprows ==0 and start ==0 and end !=0:
       dataset = file.read().splitlines()[0:end]
    if skiprows ==0 and start !=0:
       dataset = file.read().splitlines()[start:]
    if skiprows ==0 and start !=0 and end !=0:
       dataset = file.read().splitlines()[start:end]
    else:
       dataset = file.read().splitlines()
    #print (dataset)
    newdata = []
    for i in range(0,len(dataset)):
        line = dataset[i]
        if line[0] != comment:
           temp = line.split(delimiter2,target_col)
           #print(temp)
           feature = temp[target_col]
           label = temp[0]
           #print(label)
           if label == 'SVM':
               label = 0
           if label == 'N':
               label = 0
           fea = feature.split(delimiter2)
           newline = []
           #newline.append(int(label))
           newline.append(label)
           for j in range(0,len(fea)):
               if fea[j].find(':') >0 :
                   (num,val) = fea[j].split(':')
                   newline.append(float(val))
            
           newdata.append(newline)
    data = np.array(newdata, dtype=dtype)
    file.close()
    return data
def file_name(file_dir):
  L=[]
  for root, dirs, files in os.walk(file_dir):
    for file in files:
        L.append(os.path.join(root, file))
        #L.append(file)
  return L

if __name__ == '__main__':
    data=pd.read_csv("/public/home/yels/project/test/label.csv")
    df= pd.DataFrame(data) 
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
    List=[]
    for i in range(0,length):
        B=np.empty(shape=[0,7])
        a=L[i]
        if i <length-1:
           b=L[i+1]
        else:
           b=df.shape[0]
        d1=df.loc[a].values[0]
        target=d1.split("-")[0] 
        # path='/public/home/yels/project/CASP/CASP11/casp11_feature_s2/Local/feature/'+target
        #os.mkdir(path)
        for j in range(a,b):
            d1=df.loc[j].values[0]
            d2=float(df.loc[j].values[1])
            target=d1.split("-")[0] 
            if len(target)>=7:
               name=d1[8:]
            else:
               name=d1[6:]
            try:
                #/public/home/yels/project/test/rosetta/T0949_out/T0949/model1.pdb.features.lowres_global.resetta.svm
                # dir1='/public/home/yels/project/test/rosetta/'+target+'_out'
                # rosetta_global=dir1+'/ALL_scores/rosetta/'+target+'/'+name+'.pdb.features.lowres_global.resetta.svm'
                dir1='/public/home/yels/project/test/rosetta/'+target+'_out'+'/'+target
                rosetta_global=dir1+'/'+name+'.pdb.features.lowres_global.resetta.svm'
                temp = import_DLS2FSVM1(rosetta_global,delimiter=' ') #120
                rosetta=temp[0][1:]
                rosetta=rosetta.reshape(1,7)
                B=np.vstack((B,rosetta))
            except:
                print(target,name)   
        print(B.shape)
        path='/public/home/yels/project/test/rosetta_out/'+target+'.npy'
        np.save(path,B)