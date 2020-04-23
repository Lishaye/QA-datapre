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
from Bio.PDB.PDBParser import PDBParser
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
def import_DLS2FSVM2(filename, delimiter='\t', delimiter2=' ',comment='>',skiprows=0, start=0, end = 0,target_col = 1, dtype=np.float32):
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
           temp = line.split(delimiter,target_col)
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
def import_DLS2FSVM3(filename, delimiter='\t', delimiter2=' ',comment='>',skiprows=0, start=0, end = 0,target_col = 1, dtype=np.float32):
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
        print(line[0])
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
          # newline.append(label)
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
########### SS_match SA_match disfeature(label) have been numpy#######################
    data=pd.read_csv("/public/home/yels/project/CASP/CASP11/casp11_label/global_s1.csv")
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
    count=0
    for i in range(0,length):
        a=L[i]
        if i <length-1:
           b=L[i+1]
        else:
           b=df.shape[0]
        d1=df.loc[a].values[0]
        target=d1.split("-")[0] 
        path='/public/home/yels/project/CASP/CASP11/casp11_feature_s1/Local/feature/'+target
        #os.mkdir(path)
        for j in range(a,b):
            d1=df.loc[j].values[0]
            d2=float(df.loc[j].values[1])
            target=d1.split("-")[0] 
            if len(target)>=7:
               name=d1[8:]
            else:
               name=d1[6:]
            path='/public/home/yels/project/CASP/CASP11/casp11_stage1/'+target+'/'+name
            # df1=get_pdb("1",path)
            # L1=[]
            # for j in range(0,df1.shape[0]):
                # d1=df1.loc[j].values[1]
                # L1.append(d1-1)
            # print(len(L1))
            dir1='/public/home/yels/project/CASP/CASP11/casp11_feature_s1/Local/deepqa/'+target+'_out'
            dir2='/public/home/yels/project/CASP/CASP11/casp11_feature_s1/Local/disfeature/'+target
            #/public/home/yels/project/CASP/CASP11/casp11_feature_s1/Local/label/
            dir3='/public/home/yels/project/CASP/CASP11/casp11_feature_s1/Local/label/'+target
            featurefile=dir1+'/ALL_scores/'+target+'.fea_aa_ss_sa'
            pssmfile=dir1+'/ALL_scores/'+target+'.pssm_fea'
            Feature_disorder=dir1+'/ALL_scores/'+target+'.disorder_label'
            rosetta_highfile=dir1+'/ALL_scores/rosetta/'+target+'/'+name+'.pdb.features.highres.resetta.svm'
            rosetta_lowfile=dir1+'/ALL_scores/rosetta/'+target+'/'+name+'.pdb.features.lowres.resetta.svm'
            SS_matchfile=dir1+'/SS_Match/'+name+'.npy' 
            SA_matchfile=dir1+'/SA_Match/'+name+'.npy'
            disfeature=dir2+'/'+name+'.npy' ##最后一列是label
            label=dir3+'/'+name+'.npy'
            
            # featuredata = import_DLS2FSVM2(featurefile) ##25
            # pssmdata = import_DLS2FSVM2(pssmfile)       ##20
            # disorderdata = import_DLS2FSVM1(Feature_disorder,delimiter=' ')     #1
            # rosetta_highdata = import_DLS2FSVM1(rosetta_highfile,delimiter=' ') #120
            # rosetta_lowdata = import_DLS2FSVM1(rosetta_lowfile,delimiter=' ')   #35
            # ss=np.load(SS_matchfile,allow_pickle=True)
            # sa=np.load(SA_matchfile,allow_pickle=True)
            # ss_sa=np.concatenate((ss,sa),axis=1)
            # print(target,name)
            # print(ss_sa.shape)
            # disfeature=np.load(disfeature)
            # print(disfeature.shape)
            # pssm_fea = pssmdata[:,1:]
            # disorder_fea = disorderdata[:,1:]
            # fea_len = int((featuredata.shape[1]-1)/(20+3+2))
            # model_len=disfeature.shape[0]
            # print( fea_len)
            # print(model_len)
            # b=np.zeros((model_len,2))##存储SS_SA
            # b[:ss_sa.shape[0],:]=ss_sa
            # train_ss_sa=b
            # train_feature = featuredata[:,1:]
            # train_feature_seq = train_feature.reshape(fea_len,25)
            # train_feature_aa = train_feature_seq[:,0:20]
            # train_feature_ss = train_feature_seq[:,20:23]
            # train_feature_sa = train_feature_seq[:,23:25]
            # train_feature_pssm = pssm_fea.reshape(fea_len,20)
            # train_feature_disorder=disorder_fea.reshape(fea_len,1)
            # train_rosetta_highdata = rosetta_highdata[:,1:121] #(91L, 120L)  1:25 only score,no win
            # train_rosetta_lowdata = rosetta_lowdata[:,1:36] #(91L, 35L)\
            # train_feature_rosetta_highdata=train_rosetta_highdata.reshape(model_len,120)
            # train_feature_rosetta_lowdata=train_rosetta_lowdata.reshape(model_len,35)
            
            # min_pssm=-8
            # max_pssm=16
            # train_feature_pssm_normalize = np.empty_like(train_feature_pssm)
            # train_feature_pssm_normalize[:] = train_feature_pssm
            # train_feature_pssm_normalize=(train_feature_pssm_normalize-min_pssm)/(max_pssm-min_pssm)

            # featuredata_fasta = np.concatenate((train_feature_aa,train_feature_ss,train_feature_sa,train_feature_pssm_normalize,train_feature_disorder), axis=1)
            # L1=[i for i in L1 if not(i>fea_len-1)]
            # print('L1:',len(L1))
            # print('featuredata_fasta.shape',featuredata_fasta.shape)
            # b1=np.empty((len(L1),46))
            # for i in range(0,len(L1)):
                # index=L1[i]
                # print(index)
                # for j in range(0,46):
                    # b1[i,j]=featuredata_fasta[index,j]
            # print('b1:',b1.shape)   
            # train_fasta=b1                
            # featuredata_local=np.concatenate((train_fasta,train_ss_sa,train_feature_rosetta_lowdata,train_feature_rosetta_highdata,disfeature), axis=1)
            # print(featuredata_local.shape)
            # List.append(featuredata_local)
            if os.path.exists(featurefile) and os.path.exists(SS_matchfile) and os.path.exists(SA_matchfile) and os.path.exists(rosetta_highfile) and os.path.exists(disfeature) and os.path.exists(label) :
               count=count+1
            else:
                print(target,name)
    print(count)
            # try:
                # if os.path.exists(featurefile) and os.path.exists(SS_matchfile) and os.path.exists(SA_matchfile) and os.path.exists(rosetta_highfile) and os.path.exists(disfeature):
                    # featuredata = import_DLS2FSVM2(featurefile) ##25
                    # pssmdata = import_DLS2FSVM2(pssmfile)       ##20
                    # disorderdata = import_DLS2FSVM1(Feature_disorder,delimiter=' ')     #1
                    # rosetta_highdata = import_DLS2FSVM1(rosetta_highfile,delimiter=' ') #120
                    # rosetta_lowdata = import_DLS2FSVM1(rosetta_lowfile,delimiter=' ')   #35
                    # ss=np.load(SS_matchfile,allow_pickle=True)
                    # sa=np.load(SA_matchfile,allow_pickle=True)
                    # ss_sa=np.concatenate((ss,sa),axis=1)
                    # print(target,name)
                    # print(ss_sa.shape)
                    # disfeature=np.load(disfeature)
                    # print(disfeature.shape)
                    # pssm_fea = pssmdata[:,1:]
                    # disorder_fea = disorderdata[:,1:]
                    # fea_len = int((featuredata.shape[1]-1)/(20+3+2))
                    # model_len=disfeature.shape[0]
                    # print( fea_len)
                    # print(model_len)
                    # b=np.zeros((model_len,2))##存储SS_SA
                    # b[:ss_sa.shape[0],:]=ss_sa
                    # train_ss_sa=b
                    # train_feature = featuredata[:,1:]
                    # train_feature_seq = train_feature.reshape(fea_len,25)
                    # train_feature_aa = train_feature_seq[:,0:20]
                    # train_feature_ss = train_feature_seq[:,20:23]
                    # train_feature_sa = train_feature_seq[:,23:25]
                    # train_feature_pssm = pssm_fea.reshape(fea_len,20)
                    # train_feature_disorder=disorder_fea.reshape(fea_len,1)
                    # train_rosetta_highdata = rosetta_highdata[:,1:121] #(91L, 120L)  1:25 only score,no win
                    # train_rosetta_lowdata = rosetta_lowdata[:,1:36] #(91L, 35L)\
                    # train_feature_rosetta_highdata=train_rosetta_highdata.reshape(model_len,120)
                    # train_feature_rosetta_lowdata=train_rosetta_lowdata.reshape(model_len,35)
                    
                    # min_pssm=-8
                    # max_pssm=16
                    # train_feature_pssm_normalize = np.empty_like(train_feature_pssm)
                    # train_feature_pssm_normalize[:] = train_feature_pssm
                    # train_feature_pssm_normalize=(train_feature_pssm_normalize-min_pssm)/(max_pssm-min_pssm)

                    # featuredata_fasta = np.concatenate((train_feature_aa,train_feature_ss,train_feature_sa,train_feature_pssm_normalize,train_feature_disorder), axis=1)
                    # L1=[i for i in L1 if not(i>fea_len-1)]
                    # print('L1:',len(L1))
                    # print('featuredata_fasta.shape',featuredata_fasta.shape)
                    # b1=np.empty((len(L1),46))
                    # for i in range(0,len(L1)):
                        # index=L1[i]
                        # print(index)
                        # for j in range(0,46):
                            # b1[i,j]=featuredata_fasta[index,j]
                    # print('b1:',b1.shape)   
                    # train_fasta=b1                
                    # featuredata_local=np.concatenate((train_fasta,train_ss_sa,train_feature_rosetta_lowdata,train_feature_rosetta_highdata,disfeature), axis=1)
                    # print(featuredata_local.shape)
                    # List.append(featuredata_local)
            # except: 
                # print("wrong")
         
    # print(len(List))
    # np.save('/public/home/yels/project/CASP/CASP11/casp11_feature_s2/Local/feature/local.npy',List)
        