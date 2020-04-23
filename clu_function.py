# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch #用于进行层次聚类，画层次聚类图的工具包
import scipy.spatial.distance as ssd
from scipy.cluster.vq import vq,kmeans,whiten
import scipy
import heapq
def MkIndex(num_list,topk):       
    max_number = heapq.nlargest(topk, num_list) 
    max_index = []
    for t in max_number:
        index =num_list.index(t)
        max_index.append(index)
        num_list[index] = 0
    return  max_index
#####--------1类内模型个数------------
def clu_f1(mtx):
    disMat=mtx
    d=1-disMat ##相似度是越接近1越好，这样d越接近0越好
    L=[]
    list=['single','complete','average','centroid']
    for l in list :
        Z=sch.linkage(d,method=l)
        if l=='single':
           cutoff=float(0.2)
        elif l=='complete':
           cutoff=float(0.4)
        elif l=='average':
           cutoff=float(0.3)
        else:  
           cutoff=float(0.25)
        cluster= sch.fcluster(Z, t=cutoff,criterion='distance') ##相似度小于0.5的合并为1类
        #print ("Original cluster by hierarchy clustering:\n",cluster)##cluster输出为1维矩阵，相同数字的代表在同一类
        dim =cluster.shape[0]
        l=[]
        for i in range(0,dim):
            s1=cluster[i]
            c=0
            for j in range(0,dim):
                s2=cluster[j]
                if s1==s2 and i!=j:
                   c=c+1
            s=c/(dim-1)
            l.append(s)
        arry=np.array(l)
        arry=arry.reshape((dim,1))
        #print(arry)
        L.append(arry)
    arr=np.array(L)
    arr=arr.reshape((arr.shape[0],arr.shape[1]))
    arr=arr.T
    return arr
######--------2类内平均GDT_TS----------
def clu_f2(mtx):
    disMat=mtx
    d=1-disMat ##相似度是越接近1越好，这样d越接近0越好
    L=[]
    list=['single','complete','average','centroid']
    for l in list :
        Z=sch.linkage(d,method=l)
        if l=='single':
           cutoff=float(0.2)
        elif l=='complete':
           cutoff=float(0.4)
        elif l=='average':
           cutoff=float(0.3)
        else:  
           cutoff=float(0.25)
        cluster= sch.fcluster(Z, t=cutoff,criterion='distance') ##相似度小于0.5的合并为1类
        #print ("Original cluster by hierarchy clustering:\n",cluster)##cluster输出为1维矩阵，相同数字的代表在同一类
        dim =cluster.shape[0]
        l=[]
        for i in range(0,dim):
            s1=cluster[i]
            c=0
            sum=0
            for j in range(0,dim):
                s2=cluster[j]
                if s1==s2 and i!=j:
                   index=int(i*(dim-0.5*i-1.5)+j-1)
                   sum=sum+disMat[index]
                   c=c+1
            s=sum/(dim-1)
            l.append(s)
        arry=np.array(l)
        arry=arry.reshape((dim,1))
        #print(arry)
        L.append(arry)
    arr=np.array(L)
    arr=arr.reshape((arr.shape[0],arr.shape[1]))
    arr=arr.T
    return arr
# ####------------与类中"最好模型的"GDT_TS-----------
# def clu_f3(path):
    # disMat=np.load(path)
    # d=1-disMat ##相似度是越接近1越好，这样d越接近0越好
    # L=[]
    # list=['single','complete','average','centroid']
    # for l in list :
        # Z=sch.linkage(d,method=l)
        # if l=='single':
           # cutoff=float(0.2)
        # elif l=='complete':
           # cutoff=float(0.4)
        # elif l=='average':
           # cutoff=float(0.3)
        # else:  
           # cutoff=float(0.25)
        # cluster= sch.fcluster(Z, t=cutoff,criterion='distance') ##相似度小于cutoff的合并为1类
        # dim =cluster.shape[0]
        # b=np.empty((dim,dim))
        # for i in range(0,dim):
            # for j in range (i,dim):
                # if i==j:
                   # b[i][j]=1
                # else:
                   # index=int(i*(dim-0.5*i-1.5)+j-1)
                   # b[i][j]=disMat[index]
                   # b[j][i]=b[i][j]
        # S=[]#存储每个model的GDT_TS总和
        # for i in range(0,dim):
            # s=0
            # for j in range (0,dim):
                # s=s+b[i,j]
            # S.append(s)  
        # LL=[]
        # for i in range(0,dim):
            # s1=cluster[i]
            # L1=[] ##存储属于一类的索引
            # for j in range(0,dim):
                # s2=cluster[j]
                # if s1==s2 and i!=j:
                   # L1.append(j)
            # dim1=len(L1)
            # for i in range(0,dim1):
                # L2=[]
                # index=L1[i]
                # s=S[index]
                # L2.append(s)
            # n=L2.index(max(L2)) ##最好模型的索引
            # s=b[i,n]
            # LL.append(s)
        # arry=np.array(LL)
        # arry=arry.reshape((dim,1))
        # L.append(arry)
    # arr=np.array(L)
    # arr=arr.reshape((arr.shape[0],arr.shape[1]))
    # arr=arr.T
    # return arr

#####--------4与其他模型的平均相似性------------
def clu_f4(mtx):
    disMat=mtx
    d=1-disMat ##相似度是越接近1越好，这样d越接近0越好
    Z=sch.linkage(d,method='single')
    dim=Z.shape[0]+1
    b=np.empty((dim,dim))
    L1=[]
    for i in range(0,dim):
        for j in range (i,dim):
            if i==j:
               b[i][j]=1
            else:
               index=int(i*(dim-0.5*i-1.5)+j-1)
               b[i][j]=disMat[index]
               b[j][i]=b[i][j]
    for i in range(0,dim):
        sum=0
        for j in range(0,dim):
            sum=sum+b[i,j]
        L1.append((sum-1)/(dim-1))
    arry=np.array(L1)
    arry=arry.reshape((dim,1))
    return arry  


    
#####--------5与最好的模型的相似性------------
def clu_f5(mtx):
    disMat=mtx
    d=1-disMat ##相似度是越接近1越好，这样d越接近0越好
    Z=sch.linkage(d,method='single')
    dim=Z.shape[0]+1
    b=np.empty((dim,dim))
    L1=[]
    for i in range(0,dim):
        for j in range (i,dim):
            if i==j:
               b[i][j]=1
            else:
               index=int(i*(dim-0.5*i-1.5)+j-1)
               b[i][j]=disMat[index]
               b[j][i]=b[i][j]
    for i in range(0,dim):
        sum=0
        for j in range(0,dim):
            sum=sum+b[i,j]
        L1.append(sum)
    n=L1.index(max(L1))
    arry=b[:,n]  
    arry=arry.reshape((dim,1))
    return arry  

#####--------6与top n模型的相似性------------
def clu_f6(mtx):
    disMat=mtx
    d=1-disMat ##相似度是越接近1越好，这样d越接近0越好
    Z=sch.linkage(d,method='single')
    dim=Z.shape[0]+1
    b=np.empty((dim,dim))
    L1=[]
    for i in range(0,dim):
        for j in range (i,dim):
            if i==j:
               b[i][j]=1
            else:
               index=int(i*(dim-0.5*i-1.5)+j-1)
               b[i][j]=disMat[index]
               b[j][i]=b[i][j]
    for i in range(0,dim):
        sum=0
        for j in range(0,dim):
            sum=sum+b[i,j]
        L1.append(sum)
    max_index=MkIndex(L1,20)
    #print(max_index)
    arry=np.zeros(dim)
    for i in max_index:
        arry+=b[:,i]/20
        #print(b[:,i])
    arry=arry.reshape((dim,1))
    #print(arry)
    return arry  
def cluster(mtx):
    s1=clu_f1(mtx)
    s2=clu_f2(mtx)
    s4=clu_f4(mtx)
    s5=clu_f5(mtx)
    s6=clu_f6(mtx)
    C=np.hstack((s1,s2))
    C=np.hstack((C,s4))
    C=np.hstack((C,s5))
    C=np.hstack((C,s6))
    return C
  
if __name__ == '__main__':
    path='/public/home/yels/project/global_code/T0709_out/s1/clu/mtx.npy'
    C=cluster(path)
    print(C)
    print(C.shape)
    np.save('/public/home/yels/project/global_code/T0709_out/s1/clu/clu.npy',C)

     
