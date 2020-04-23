# -*- coding: utf-8 -*-
import os
import subprocess
import numpy as np
import pandas as pd
import re,sys
import os, shutil
import numpy as np
from clu_function import*
def file_name(file_dir):  
  L=[]  
  for root, dirs, files in os.walk(file_dir): 
    for file in files: 
        if os.path.splitext(file)[1] != '.inp':
           L.append(os.path.join(root, file))
  return L
def clufeature(target):
    Models=file_name(target)
    Name=[]
    for model in Models:
        Name.append(model)
    length=len(Name)
    l=[]
    for i in range(length-1):
        M1=Name[i]
        for j in range (i+1,length):
            M2=Name[j]
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
            # try:
                # arry=np.array(list)
                # print(arry)
                # lines=arry.astype(dtype=np.float)
                # lines[0]=round(lines[0],3)
                # print(lines[0])
                # l.append(lines[0])
            # except:
                # agrs='g++'+' '+'/public/home/yels/project/Testmodel/TMscore.cpp'
                # Temp=os.system(agrs)             
    mtx=np.array(l)  
    return mtx
target=sys.argv[1] ##pdb
outdir= sys.argv[2] ##output
clu=outdir+'/'+'clu'

if os.path.exists(outdir):
    print("outdir have exist")
else:
    os.mkdir(outdir)
if os.path.exists(clu):
    print("clu have exist")
else:
    os.mkdir(clu)
agrs='g++'+' '+'/public/home/yels/project/Testmodel/TMscore.cpp'
Temp=os.system(agrs)
print(Temp)
mtx=clufeature(target)
print(mtx.shape)
C=cluster(mtx)
print(C.shape)
path1=clu+'/'+'clu.npy'
np.save(path1,C)
Models=file_name(target)
Models.sort()
Name=[]
for model in Models:
    name=model.split('/')[-1]
    print(name)
    Name.append(name)
path2=clu+'/'+'index.npy'
np.save(path2,Name)
print(len(Name))