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
import re,sys
p = PDBParser(PERMISSIVE=1)
def file1_name(file_dir):  
  L=[]  
  for root, dirs, files in os.walk(file_dir): 
    for file in files: 
      if os.path.splitext(file)[1] == '.pdb': 
        L.append(os.path.join(root, file))
        #L.append(file)
  return L
def file2_name(file_dir):  
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
                if residue.has_id('CA'):         
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
                else:
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
                    
                    
    df_all=pd.concat([df1,df2,df3],axis=1) 
    return(df_all)
if __name__ == '__main__':
    
    native=sys.argv[1] ##/public/home/yels/project/CASP/CASP12/casp12_target/
    Model=sys.argv[2] ##/public/home/yels/project/CASP/CASP12/casp12_stage2/
    outfile= sys.argv[3] ##/public/home/yels/project/CASP/CASP12/casp12_label/dists_s2.csv
    
    file=open(outfile,'a',newline='')
    csv_write = csv.writer(file,dialect='excel')
    N=['index','seq']
    csv_write.writerow(N)
    f1=file1_name(native)
    for f in f1:
    #/public/home/yels/project/CASP/CASP10/casp10_target/T0648-D1.pdb
        temp=f.split('/')[-1]
        target=temp.split('.')[0]
        if len(target)>5:
           target=target.split('-')[0]
        print(target)
        f2=Model+target
        df2=get_pdb("1",f)
        ff2=file2_name(f2)
        for ff in ff2:
            name=ff.split('/')[-1]
            df1=get_pdb("1",ff)
            args='./a.out'+' '+ff+' '+f
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
            print(arry)
            lines=arry.astype(dtype=np.float)
            N=[]
            # model=target+'-'+name
            # N.append(model)
            # print(lines)
            # N.append(lines[0])
            # csv_write.writerow(N)
            # file1.close()   
            #print(df1)
            #print(lines)
            df1['x1']=df1.apply(lambda x:x['x']*lines[2]+x['y']*lines[3]+x['z']*lines[4]+lines[1],axis=1)
            df1['y1']=df1.apply(lambda x:x['x']*lines[6]+x['y']*lines[7]+x['z']*lines[8]+lines[5],axis=1)
            df1['z1']=df1.apply(lambda x:x['x']*lines[10]+x['y']*lines[11]+x['z']*lines[12]+lines[9],axis=1)
            #print(df1)
            result=pd.merge(df1,df2,on=['atom','chain'],how='inner')
            result.eval('dists=sqrt((x1-x_y)**2+(y1-y_y)**2+(z1-z_y)**2)',inplace=True)
            #print(result)
            df1=df1.set_index(['atom','chain'])
            df1['dists']=result.set_index(['atom','chain'])['dists']
            df1=df1.reset_index()
            df1=df1.fillna(-1)
            #print(df1)
            # file=open("/public/home/yels/project/CASP/CASP10/casp10_label/dists_s11.csv",'a',newline='')
            # csv_write = csv.writer(file,dialect='excel')
            L=[]
            L.append(target+'-'+ff.split("/")[-1])
            L.append(df1['dists'].shape[0])
            #print(len(L))
            for dist in df1['dists']:   
                a = ("%.6f" % dist)
                L.append(a)
            print(len(L))
            csv_write.writerow(L)
    file.close()
                # print(n+'-'+ff.split("/")[-1])     