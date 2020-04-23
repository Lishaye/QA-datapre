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
def file_name(file_dir):
  L=[]
  for root, dirs, files in os.walk(file_dir):
    for file in files:
        L.append(os.path.join(root, file))
        #L.append(file)
  return L
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
    L=['T0838','T0823','T0835','T0765']
    for target in L:
        path='/public/home/yels/project/local_code/test/'+target+'_out/disfeature'
        os.mkdir(path)
        #/public/home/yels/project/CASP/CASP11/casp11_stage1/T0759/server01_TS1
        dir='/public/home/yels/project/CASP/CASP11/casp11_stage1/'+target
        file=file_name(dir)
        for f in file:
        ##输入1pdb 2model的距离矩阵 3native预测的距离信息
            name=f.split("/")[-1]
            df1=get_pdb("1",f)
            L1=[]
            for j in range(0,df1.shape[0]):
                d1=df1.loc[j].values[1]
                L1.append(d1-1)
            #print(L1)
            print(len(L1))
            path1='/public/home/yels/project/CASP/CASP11/casp11_feature_s1/distance/Dis/'+target+'/'+name+'.npz.npy'
            path2='/public/home/yels/project/CASP/CASP11/casp11_feature_s1/distance/npz/'+target+'.npz' 
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
            path3=path+'/'+name+'.npy'
            np.save(path3,feature)
       
         
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
 
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




