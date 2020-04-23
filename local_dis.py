# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:50:08 2019

@author: 32420
"""
import os 
import os
import subprocess
import numpy as np
import pandas as pd
import math 
from Bio.PDB.PDBParser import PDBParser
import csv
from Disfeature import *
p = PDBParser(PERMISSIVE=1)
def get_pdb(structure_id,filename):
    s = p.get_structure(structure_id, filename)
    df1= pd.DataFrame({'atom':[]},dtype='str')
    df2=pd.DataFrame({'chain':[]},dtype='int')
    df3 = pd.DataFrame({'x':[],'y':[],'z':[]},dtype='float')
    df_all=pd.DataFrame([])
    for model in s.get_list():
        for chain in model.get_list():
            for residue in chain.get_list():
                if residue.has_id('CB'):         
                    for atom in residue:
                        if atom.name=='CB':
                            residue_id = residue.get_id()
                            hetfield = residue_id[1]
                            dff=pd.DataFrame([hetfield],columns=df2.columns)
                            df2=pd.concat([df2,dff],ignore_index=True)
                            df=pd.DataFrame([atom.get_id()],columns=df1.columns)
                            df1=pd.concat([df1,df],ignore_index=True)
                            dfff = pd.DataFrame(atom.get_coord().reshape(1,3),columns=df3.columns)
                            df3 = pd.concat([df3,dfff],ignore_index=True)
                else:
                    for atom in residue:                
                        if atom.name=='CA':
                            residue_id = residue.get_id()
                            hetfield = residue_id[1]
                            dff=pd.DataFrame([hetfield],columns=df2.columns)
                            df2=pd.concat([df2,dff],ignore_index=True)
                            df=pd.DataFrame([atom.get_id()],columns=df1.columns)
                            df1=pd.concat([df1,df],ignore_index=True)
                            dfff = pd.DataFrame(atom.get_coord().reshape(1,3),columns=df3.columns)
                            df3 = pd.concat([df3,dfff],ignore_index=True)
                    
                    
    df_all=pd.concat([df1,df2,df3],axis=1) 
    return(df_all)
if __name__ == '__main__': 
    data=pd.read_csv("/public/home/yels/project/CASP/CASP10/casp10_label/dists_s2.csv")
    df= pd.DataFrame(data) 
    print(df.shape[0])
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
    L=L[60:]
    print(L)
    length=len(L)
    print(length)
    for i in range(0,length):
        B=np.empty(shape=[0,15])
        List=[]
        a=L[i]
        if i <length-1:
           b=L[i+1]
        else:
           b=df.shape[0]
        d1=df.loc[a].values[0]
        target=d1.split("-")[0] 
        # path='/public/home/yels/project/CASP/CASP10/casp10_feature_s1/Local/disfeature/'+target
        # os.mkdir(path)
        for j in range(a,b):
            Label=[]
            d1=df.loc[j].values[0]
            target=d1.split("-")[0] 
            if len(target)>=7:
               name=d1[8:]
            else:
               name=d1[6:]
            dim=df.loc[j].values.shape[0]
            for k in range(2,dim):   
                x = float(df.loc[j].values[k])
                if math.isnan(x):
                  continue
                else:
                  Label.append(x)
            a=np.array(Label)
            arry=a.reshape(a.shape[0],1)##label
            ##输入1pdb 2model的距离矩阵 3native预测的距离信息
            path='/public/home/yels/project/CASP/CASP10/casp10_stage2/'+target+'/'+name
            df1=get_pdb("1",path)
            L1=[]
            for j in range(0,df1.shape[0]):
                d1=df1.loc[j].values[1]
                L1.append(d1-1)
            #print(L1)
            print(len(L1))
            path1='/public/home/yels/project/CASP/CASP10/casp10_feature_s2/distance/Dis/'+target+'/'+name+'.npz.npy'
            path2='/public/home/yels/project/CASP/CASP10/casp10_feature_s1/distance/npz/'+target+'.npz' 
            mtx1=np.load(path1)##增加判断
            mtx21=M_mtx(path2) ##最大距离矩阵
            mtx22=W_mtx(path2)  ##加权距离矩阵
            dim=mtx1.shape[0]-mtx21.shape[0]
            if dim>0:
               mtx1=mtx1[:mtx21.shape[0],:mtx21.shape[0]]
            print(mtx1.shape)
            print(mtx21.shape)
            print(mtx22.shape)
            b1=np.empty((mtx1.shape[0],mtx1.shape[0]))
            b2=np.empty((mtx1.shape[0],mtx1.shape[0]))
            for i in range(0,len(L1)):
                index=L1[i]
                for j in range(0,len(L1)):
                    b1[i,j]=mtx21[index,j]
                    b2[i,j]=mtx22[index,j]
            print(b1.shape)
            print(b2.shape)
            feature1=D_Fun(mtx1,b1)
            feature2=D_Fun(mtx1,b2)
            feature=np.hstack((feature1,feature2))
            print('%s-%s:'%(target,name),feature.shape)
            c=np.hstack((feature,arry))
            print(c.shape)
            B=np.vstack((B,c))
            path4='/public/home/yels/project/CASP/CASP10/casp10_feature_s2/Local/disfeature/'+target+'/'+name+'.npy'
            np.save(path4,c)
        path5='/public/home/yels/project/CASP/CASP10/casp10_feature_s2/Local/Disfeature/'+target+'.npy'
        print(B.shape)
        np.save(path5,B)
         
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
 
    # df=pd.read_excel("/public/home/yels/project/Distance_feature/test2.xlsx")
    # for j in range(0,4):
        # L = []
        # d1=df.loc[j].values[0]
        # target=d1.split("-")[0] 
        # if len(target)>=7:
           # name=d1[8:]
        # else:
           # name=d1[6:]
        # dim=df.loc[j].values.shape[0]
        # for i in range(2,dim):   
            # x = float(df.loc[j].values[i])
            # if math.isnan(x):
              # continue
            # else:
              # L.append(x)
        # a=np.array(L)
        # arry=a.reshape(a.shape[0],1)##label
        # ##输入1pdb 2model的距离矩阵 3native预测的距离信息
        # path='/public/home/yels/project/CASP/CASP10/casp10_stage1/'+target+'/'+name
        # df1=get_pdb("1",path)
        # L1=[]
        # for j in range(0,df1.shape[0]):
            # d1=df1.loc[j].values[1]
            # L1.append(d1-1)
        # #print(L1)
        # print(len(L1))
        # path1='/public/home/yels/project/CASP/CASP10/casp10_feature_s2/distance/Dis/'+target+'/'+name+'.npz.npy'
        # path2='/public/home/yels/project/CASP/CASP10/casp10_feature_s2/distance/npz/'+target+'.npz' 
        # mtx1=np.load(path1)##增加判断
        # mtx21=M_mtx(path2) ##最大距离矩阵
        # mtx22=W_mtx(path2)  ##加权距离矩阵
        # dim=mtx1.shape[0]-mtx21.shape[0]
        # if dim>0:
           # mtx1=mtx1[:mtx21.shape[0],:mtx21.shape[0]]
        # print(mtx1.shape)
        # print(mtx21.shape)
        # print(mtx22.shape)
        # b1=np.empty((mtx1.shape[0],mtx1.shape[0]))
        # b2=np.empty((mtx1.shape[0],mtx1.shape[0]))
        # for i in range(0,len(L1)):
            # index=L1[i]
            # for j in range(0,len(L1)):
                # b1[i,j]=mtx21[index,j]
                # b2[i,j]=mtx22[index,j]
        # print(b1.shape)
        # print(b2.shape)
        # feature1=D_Fun(mtx1,b1)
        # feature2=D_Fun(mtx1,b2)
        # feature=np.hstack((feature1,feature2))
        # print('%s-%s:'%(target,name),feature.shape)
        # c=np.hstack((feature,arry))
        # print(c.shape)
        # path4='/public/home/yels/project/CASP/CASP10/casp10_feature_s2/Local/disfeature/'
        # np.save(path4,c)
          
    # ###测试特征相关系数
    # df=pd.read_excel("/public/home/yels/project/Distance_feature/test2.xlsx")
    # for j in range(3,4):
        # L = []
        # model=df.loc[j].values[0]
        # dim=df.loc[j].values.shape[0]
        # for i in range(2,dim):   
            # x = float(df.loc[j].values[i])
            # if math.isnan(x):
              # continue
            # else:
              # L.append(x)
        # a=np.array(L)
        # arry=a.reshape(a.shape[0],1)
        # #print(arry)
        # print('label:',arry.shape)
        # C=np.hstack((arry,feature))
        # C=C.T
        # data=np.corrcoef(C)
        # data=np.around(data, decimals=4)
        # data=data.tolist()
        # print("sum person:",data[0])




