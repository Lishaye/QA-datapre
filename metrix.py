import numpy as np
import pandas as pd
import math 
def pearson_r(y_true, y_pred):
    y_true=y_true.reshape(y_true.shape[0],1)
    y_pred=y_pred.reshape(y_pred.shape[0],1)
    C=np.hstack((y_true,y_pred))
    C=C.T
    data=np.corrcoef(C)
    data=np.around(data, decimals=4)
    data=data.tolist()
    corr=data[0][1]
    return corr
def best_different(y_true, y_pred):
    p=y_pred.tolist()
    t=y_true.tolist()
    index1=p.index(max(p))
    index2=t.index(max(t))
    diff=abs(t[index1][0]-t[index2][0])*100
    return diff
def target_loss(y_true, y_pred):
    x=y_true*100
    y=y_pred*100
    z=np.mean(abs(x-y))
    return z
def ASE(y_true, y_pred):
    d1 = y_true
    d2 = y_pred
    x1 = 1 + np.square(d1/3)
    x2 = 1 + np.square(d2/3)
    s1 = 1/x1
    s2 = 1/x2
    ASE = 100*(1-np.mean(abs(s1-s2)))
    return ASE
    
    
    
    
if __name__ == '__main__':
   y_true=np.array([[3, 6]])
   y_pred=np.array([[6, 9]])
   y_true=y_true.T 
   y_pred=y_pred.T
   ASE=ASE(y_true, y_pred)
   print(ASE)