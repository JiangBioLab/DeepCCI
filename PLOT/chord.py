import numpy as np
import pandas as pd
import ast
import operator
from itertools import chain,groupby
import math
import os
from scipy import sparse
from collections import defaultdict
if not os.path.exists("./output/chord"):
    os.system('mkdir ./output/chord')
if not os.path.exists("./output/chord/File"):
    os.system('mkdir ./output/chord/File')
if not os.path.exists("./output/chord/Plot"):
    os.system('mkdir ./output/chord/Plot')
CC_net = pd.read_csv('../INTERACTION/output/CCI_out.csv',header=None).values
CC_net_data = CC_net[1:,1:]
CC_pval = CC_net_data[:,5].astype(np.float16)
source = CC_net_data[:,0]
target = CC_net_data[:,1]
for i in range(len(source)):
    source[i] = source[i].replace(" ","")
    target[i] = target[i].replace(" ","")

Cell_type_old = np.unique(source)
Cell_type = []
for i in range(len(Cell_type_old)):
    Cell_type.append(Cell_type_old[i])


matix= np.zeros((len(Cell_type),len(Cell_type)))

for i in range(len(Cell_type)):
    for j in range(len(Cell_type)):
        for k in range(len(source)):
            if source[k] == Cell_type[i] and target[k] == Cell_type[j]:
                if CC_pval[k]==1:
                    matix[i,j]+=1
pd.DataFrame(matix.astype(np.int32), index=Cell_type,columns=Cell_type).to_csv("./output/CCImatix.csv",quoting=1)

for i in range(len(Cell_type)):
    CellMatix=np.zeros((len(Cell_type),len(Cell_type)))
    CCImatix = pd.read_csv('./output/CCImatix.csv',header=None).values[1:,1:]
    CellMatix[i,:] = CCImatix[i,:]
    CellMatix[:,i] = CCImatix[:,i]
    #print(CellMatix)
    pd.DataFrame(CellMatix.astype(np.int32), index=Cell_type,columns=Cell_type).to_csv("./output/chord/File/"+Cell_type[i]+".csv",quoting=1)
