from numpy import *
import pandas as pd
import time
import numpy as np
import os
import csv
#读取csv文件，获取要计算feature的modle
data1=pd.read_csv("/public/home/yels/project/CASP/CASP11/casp11_label/global_s1.csv")
df= pd.DataFrame(data1) 
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
length=len(L)
print(length)
list=[]
for i in range(0,length):
    a=L[i]
    if i <length-1:
       b=L[i+1]
    else:
       b=df.shape[0]
    print(a,b)
    d1=df.loc[a].values[0]
    d2=float(df.loc[a].values[1])
    target=d1.split("-")[0] 
    print(target)
    ###############stage1########################
    path1='/public/home/yels/project/CASP/CASP11/casp11_feature_s1/deepqa_npy/'+target+'.npy'
    deepqa=np.load(path1)
    print('deepqa:',deepqa.shape)
    path2='/public/home/yels/project/CASP/CASP11/casp11_feature_s1/distance/Dis_feature/globals1_24.npy'
    D=np.load(path2)
    D=D.T
    D1=D[a:b,[1,4,5,7,8,9,10,13,14,16,17,18,21,22]]
    # D1=D[a:b,[0,3,4,6,7,8,9,12,13,15,16,17,20,21]]##stage1(23)
    print('disfeature:',D1.shape)
    path3='/public/home/yels/project/CASP/CASP11/casp11_feature_s1/rosetta/'+target+'.npy'
    rosetta=np.load(path3)
    print('rosetta',rosetta.shape)
    lable=D[a:b,0]
    lable=lable.reshape((b-a,1))
    feature1=np.concatenate((deepqa,D1,rosetta,lable),axis=1)
    print(feature1.shape)
    list.append(feature1)
path='/public/home/yels/project/CASP/CASP11/casp11_feature/global36/globals1.npy'
print(len(list))
np.save(path,list)

    ###############stage2########################
    # path1='/public/home/yels/project/CASP/CASP13/casp13_feature_s2/deepqa_npy/'+target+'.npy'
    # deepqa=np.load(path1)
    # print(deepqa.shape)
    # path2='/public/home/yels/project/CASP/CASP13/casp13_feature_s2/distance/Dis_feature/'+target+'.npy'
    # D=np.load(path2)
    # D=D[:,[0,3,4,6,7,8,9,12,13,15,16,17,20,21]]
    # print(D.shape)
    # path3='/public/home/yels/project/CASP/CASP13/casp13_feature_s2/rosetta/'+target+'.npy'
    # rosetta=np.load(path3)
    # lable= df['score'][a:b].values
    # lable=lable.reshape((b-a,1))
    # feature2=np.concatenate((deepqa,D,rosetta,lable),axis=1)
    # print(feature2.shape)
    # list.append(feature2)
    # # path4='/public/home/yels/project/CASP/CASP10/casp10_feature_s1/feature1_label_g/'+target+'.npy'
    # # np.save(path4,feature)
# path='/public/home/yels/project/CASP/CASP13/casp13_feature/global36/globals2.npy'
# print(len(list))
# np.save(path,list)








































    ################stage1########################
    # path1='/public/home/yels/project/CASP/CASP10/casp10_feature_s2/deepqa_npy/'+target+'.npy'
    # deepqa=np.load(path1)
    # path2='/public/home/yels/project/CASP/CASP10/casp10_feature_s2/cluster/'+target+'.npy'
    # Clu=np.load(path2)
    # path3='/public/home/yels/project/CASP/CASP10/casp10_feature_s2/distance/Dis_feature/'+target+'.npy'
    # D2=np.load(path3)
    # D2=D2[:,[0,3,4,6,7,8,9,12,13,15,16,17,20,21]]##stage2
    # path4='/public/home/yels/project/CASP/CASP10/casp10_feature_s2/rosetta/'+target+'.npy'
    # rosetta=np.load(path4)
    # lable= df['score'][a:b].values
    # lable=lable.reshape((b-a,1))
    # feature=np.concatenate((deepqa,D2,Clu,rosetta,lable),axis=1)
    # print(feature.shape)
    # list.append(feature)
# path4='/public/home/yels/project/CASP/CASP10/casp10_feature/globals2/global48.npy'
# np.save(path4,list)
    
    
    ##############singe target#############
# data1=pd.read_csv("/public/home/yels/project/test/pose.csv")
# df= pd.DataFrame(data1) 
# for j in range(0,df.shape[0]):
    # name=df.loc[j].values[0]
    # path1='/public/home/yels/project/test/deepqa/pose.npy'
    # deepqa=np.load(path1)
    # print(deepqa.shape)
    # path3='/public/home/yels/project/test/distance/disfeature/global_23.npy'
    # D=np.load(path3)
    # D=D[:,[0,3,4,6,7,8,9,12,13,15,16,17,20,21]]
    # print(D.shape)
    # feature=np.concatenate((deepqa,D),axis=1)
    # path='/public/home/yels/project/test/feature/pose.npy'
    # print(feature.shape)
    # np.save(path,feature)