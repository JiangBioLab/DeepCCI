import numpy as np
import pandas as pd
import ast
import operator
from itertools import chain,groupby
import math
import os
from scipy import sparse
from collections import defaultdict
if not os.path.exists("./output/Network"):
    os.system('mkdir ./output/Network')
if not os.path.exists("./output/Network/File"):
    os.system('mkdir ./output/Network/File')
if not os.path.exists("./output/Network/Plot"):
    os.system('mkdir ./output/Network/Plot')

columns = ["Source","Target","count"]

CC_net = pd.read_csv('../INTERACTION/output/CCI_out.csv',header=None).values
CC_net_data = CC_net[1:,1:]
CC_pval = CC_net_data[:,5].astype(np.int32)
source = []
target = []
Ligand_all = []
Receptor_all = []
cc_idx = []
for i in range(len(CC_pval)):
    if (CC_pval[i]) == 1:
        cc_idx.append(i)
CC_new = CC_net_data[cc_idx,:]

pair_name = CC_new[:,6]
source = []
target = []
prob = []
for i in range(CC_new.shape[0]):
    source.append(CC_new[i,0].replace(" ",""))
    target.append(CC_new[i,1].replace(" ",""))
    prob.append(CC_new[i,4])


pair = np.unique(pair_name)
for j in range(len(pair)):
    Source = []
    Target = []
    count = []
    for i in range(len(pair_name)):
        if str(pair_name[i]) == pair[j]:
            Source.append(str(source[i]))
            Target.append(str(target[i]))
            count.append(str(prob[i]))
    INTER1 = np.vstack((np.array(Source),np.array(Target)))
    #INTER2 = np.vstack(((INTER1),(CellYnew)))
    #INTER3 = np.vstack(((INTER2),(Probnew)))
    INTER = np.vstack((INTER1,np.array(count))).T
    index = np.arange(1,len(Source)+1)
    pd.DataFrame(INTER, index=index,columns=columns).to_csv("./output/Network/File/"+str(pair[j])+".csv",quoting=1)















