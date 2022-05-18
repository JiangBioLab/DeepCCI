import numpy as np
import pandas as pd
import ast
import operator
from itertools import chain,groupby
import math
import os
from scipy import sparse
from collections import defaultdict
if not os.path.exists("./output/heatmap"):
    os.system('mkdir ./output/heatmap')
if not os.path.exists("./output/heatmap/File"):
    os.system('mkdir ./output/heatmap/File')
if not os.path.exists("./output/heatmap/Plot"):
    os.system('mkdir ./output/heatmap/Plot')

columns = ["Cell2Cell","Interaction","prob","pval"]

Cell = []
prob = []
pval = []
pair = []
CC_net = pd.read_csv('../INTERACTION/output/CCI_out.csv',header=None).values
CC_net_data = CC_net[1:,1:]


CC_prob = CC_net_data[:,4].astype(np.float16)
CC_pval = CC_net_data[:,5].astype(np.int32)
source = []
target = []
Interaction = []

for i in range(CC_net_data.shape[0]):
    #print(i)
    
    Interaction.append(CC_net_data[i,7])

for i in range(CC_net_data.shape[0]):
    source.append(CC_net_data[i,0].replace(" ",""))
    target.append(CC_net_data[i,1].replace(" ",""))
    #Interaction.append(CC_net_data[i,7].replace("(",""))
    #Interaction.append(CC_net_data[i,7])
source = np.array(source)


Cell_type = np.unique(source)
#Cell_type = ['B','Basophil','CD14+Mono','CD1C+_Bdendriticcell','Circulatingfetalcell','CD8+T','DC','FCGR3A+Mono','NaiveT']
np.savetxt("./cell_type.csv",Cell_type,fmt="%s",delimiter=",")
#print(Cell_type)
for j in range(len(Cell_type)):
    Cell = []
    prob = []
    pair = []

    for i in range(len(source)):
        if str(source[i]) == Cell_type[j]:
            if CC_pval[i] == 1:
                Cell.append(str(source[i])+" | "+str(target[i]))
                pair.append(Interaction[i])
                prob.append(CC_prob[i])
    idx = np.argsort(-np.array(prob))
    Cellnew = np.array(Cell)[idx]
    pairnew = np.array(pair)[idx]
    probnew = np.array(prob)[idx]
    probnew = probnew
    #pvalnew = np.array(pval)[idx]
    Cell_unique = np.unique(np.array(Cellnew))
    pair_unique = np.unique(np.array(pairnew))

    matix = np.zeros((len(Cell_unique),len(pair_unique)))
    if len(Cellnew)>=200:
        #Len=150
        Len=200
        for m in range((Len)):
            for k in range(len(Cell_unique)):
                for i in range(len(pair_unique)):
                    if Cell_unique[k] == Cellnew[m] and pair_unique[i] == pairnew[m]:
                        matix[k,i] = float(probnew[m])
    else:
        #print(len(Cellnew))
        Len=len(Cellnew)
        for m in range((Len)):
            for k in range(len(Cell_unique)):
                for i in range(len(pair_unique)):
                    if Cell_unique[k] == Cellnew[m] and pair_unique[i] == pairnew[m]:
                        matix[k,i] = float(probnew[m])

    

    idx = np.argwhere(np.all(matix[..., :] == 0, axis=0))
    a2 = np.delete(matix, idx, axis=1)
    idx_all = np.arange(len(pair_unique))
    idx1 = np.delete(idx_all, idx)
    #print(idx1)
    pd.DataFrame(a2, index=Cell_unique,columns=pair_unique[idx1]).to_csv("./output/heatmap/File/"+str(Cell_type[j])+".csv",quoting=1)










