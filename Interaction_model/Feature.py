import numpy as np
import pandas as pd
import ast
import operator
from itertools import chain
import math
import os
import argparse
from utils import *
from scipy import sparse
from collections import defaultdict
from sklearn.decomposition import TruncatedSVD
import torch
def geometricMean(x):
    if np.shape(x)[0]==0:
        y=0
    else:
        y=np.exp(np.mean(np.log(x),axis=0))
        y[np.isnan(y)]=0
    return y

def computeExpr_complex(complex_input, data_use, sorted_indexC,data_rownames):
    Rsubunits=complex_input[sorted_indexC,]
    #print(Rsubunits)
    data_complex_all = np.zeros((len(sorted_indexC),np.shape(data_use)[1]))
    for i in range(len(sorted_indexC)):
        list1=[]
        list1.append(Rsubunits.tolist()[i])

        list1 = list(chain(*list1))
        while '' in list1:
            list1.remove('')
        sorted_index=[]
        #print(list1)
        for j in list1:
            if j in data_rownames.tolist():
                sorted_index.append(data_rownames.tolist().index(j))

        data_complex=geometricMean(data_use[sorted_index,])

        data_complex_all[i]=data_complex

    return data_complex_all

def cut_graph():
    net=pd.read_csv('./output/df_net.csv').values
    P_prob = net[:,5].astype(np.float32)
    P_label=net[:,6].astype(np.float32)
    Pair_name=net[:,7]

    Pair_uni = np.unique(Pair_name)

    sum_pair = []

    for j in range(len(Pair_uni)):
        sumP = 0
        for i in range(len(Pair_name)):
            if str(Pair_name[i]) == str(Pair_uni[j]) and P_label[i]<=0.01:
                sumP += P_prob[i]
        sum_pair.append(sumP)
    #print((sum_pair))
    #print(len(sum_pair))
    sum_pair = np.array(sum_pair)
    index=np.argsort(-sum_pair)
    Pair_uni = np.array(Pair_uni)
    Pair_uninew = Pair_uni[index][:100]


    pairLRsigdata = pd.read_csv('./output/pairLR_use.csv',header=None).values
    pairname = pairLRsigdata[1:,1].tolist()
    #print(len(pairname))
    idx=[]
    for i in range(len(pairname)):
        if str(pairname[i]) in Pair_uni:
            idx.append(i)
    pair_use_new = pairLRsigdata[1:,1:][idx,:]
    #print(pair_use_new.shape)

    index = pairLRsigdata[1:,0][idx]
    columns = pairLRsigdata[0,1:]
    pd.DataFrame(pair_use_new, index=index,columns=columns).to_csv("./output/pair_use_new.csv",quoting=1)


    idx1=[]
    for i in range(len(pairname)):
        if str(pairname[i]) in Pair_uninew:
            idx1.append(i)
    pair_nui_new = pairLRsigdata[1:,1:][idx1,:]
    #print(pair_nui_new.shape)

    index = pairLRsigdata[1:,0][idx1]
    columns = pairLRsigdata[0,1:]
    pd.DataFrame(pair_nui_new, index=index,columns=columns).to_csv("./output/pair_use_top.csv",quoting=1)



def Feature():
    complex_inputdata = './output/complex_input.csv'
    with open(complex_inputdata,encoding = 'gbk') as f:

        complex_inputname=np.loadtxt(complex_inputdata,str,delimiter = ",",skiprows=1)[:,0]
        #print(complex_inputname)
        for i in range(len(complex_inputname)):
            complex_inputname[i]=ast.literal_eval(complex_inputname[i])
        subunit1 = np.loadtxt(complex_inputdata,str,delimiter = ",",skiprows=1)[:,1]
        for i in range(len(subunit1)):
            subunit1[i]=ast.literal_eval(subunit1[i])
        subunit2 = np.loadtxt(complex_inputdata,str,delimiter = ",",skiprows=1)[:,2]
        for i in range(len(subunit2)):
            subunit2[i]=ast.literal_eval(subunit2[i])
        subunit3 = np.loadtxt(complex_inputdata,str,delimiter = ",",skiprows=1)[:,3]
        for i in range(len(subunit3)):
            subunit3[i]=ast.literal_eval(subunit3[i])
        subunit4 = np.loadtxt(complex_inputdata,str,delimiter = ",",skiprows=1)[:,4]
        for i in range(len(subunit4)):
            subunit4[i]=ast.literal_eval(subunit4[i])


    complex_input1=np.vstack((np.array(subunit1),np.array(subunit2)))
    complex_input2=np.vstack((np.array(subunit3),np.array(subunit4)))
    complex_input=np.vstack((complex_input1,complex_input2)).T


    pairLRsigdata = './output/pair_use_new.csv'
    geneL = pd.read_csv(pairLRsigdata).values[:,3]
    geneR = pd.read_csv(pairLRsigdata).values[:,4]
    pairname=pd.read_csv(pairLRsigdata).values[:,1]
    nLR=len(geneL)
    data_project='./output/data_project.csv'
    with open(data_project,encoding = 'gbk') as f:
        data = np.loadtxt(data_project,str,delimiter = ",",skiprows=1)[:,1:]
        data_rownames=np.loadtxt(data_project,str,delimiter = ",",skiprows=1)[:,0]
        data=(data.astype(np.float64))
        for i in range(len(data_rownames)):
            data_rownames[i]=ast.literal_eval(data_rownames[i])

    metadata=np.loadtxt('./output/meta.csv',str,delimiter = ",",skiprows=1)[:,-1]
    for i in range(len(metadata)):
        metadata[i]=ast.literal_eval(metadata[i]).replace (' ','')
    labelname = np.unique(metadata)
    idx = defaultdict(list)

    for j in range(len(labelname)):
        for i in range(len(metadata)):
            if labelname[j]== metadata[i]:
                idx[labelname[j]].append((i))
    len_idx=[]
    for i in range(len(labelname)):
        len_idx.append(len(idx[labelname[i]]))

    #net=np.loadtxt('./output/df_net.csv',str,delimiter = ",",skiprows=1)
    net=pd.read_csv('./output/df_net.csv').values
    Cell_s=net[:,1]
    Cell_t=net[:,2]
    PairL=net[:,3]
    PairR=net[:,4]
    P_label=net[:,6]
    Pair_name=net[:,7]
    Cell1 = []
    Cell2 = []
    for i in range(len(Cell_s)):
        Cell1.append(Cell_s[i].replace (' ',''))
        Cell2.append(Cell_t[i].replace (' ',''))

    pair_name_label=[]
    for i in range(len(Cell1)):
        pair_name_label.append(Cell1[i]+'_'+Cell2[i]+'_'+Pair_name[i])

    data_use = data/np.max(data)
    nC=np.shape(data_use)[1]
    data_rownames=data_rownames.tolist()
    sorted_indexL = []
    for i in geneL.tolist():
        if i in data_rownames:
            sorted_indexL.append(data_rownames.index(i))
    sorted_indexR = []
    for i in geneR.tolist():
        if i in data_rownames:
            sorted_indexR.append(data_rownames.index(i))

    data_rownames=np.array(data_rownames)
    index_singleL=np.where(np.isin(geneL,data_rownames,invert=False)==True)
    index_complexL=np.where(np.isin(geneL,data_rownames,invert=False)==False)
    index_singleR=np.where(np.isin(geneR,data_rownames,invert=False)==True)
    index_complexR=np.where(np.isin(geneR,data_rownames,invert=False)==False)
    dataL1 = data_use[sorted_indexL,]


    dataL=np.zeros((nLR,nC))
    dataL[index_singleL,] = dataL1
    if len(index_complexL[0]) > 0:
        complexL = geneL[index_complexL]
        sorted_indexCL=[]
        complexLnewL=[]

        for i in complexL.tolist():
            if i in complex_inputname.tolist():
                sorted_indexCL.append(complex_inputname.tolist().index(i))
                complexLnewL=complexLnewL+[j for j,v in enumerate(geneL.tolist()) if v==i]
        
        complexLnewL = list(set(complexLnewL))
        complexLnewL.sort()
        data_complex=computeExpr_complex(complex_input, data_use, sorted_indexCL,data_rownames)
        dataL[complexLnewL,] = data_complex


    dataR1 = data_use[sorted_indexR,]
    dataR=np.zeros((nLR,nC))
    dataR[index_singleR,] = dataR1

    if len(index_complexR[0]) > 0:
        complexR = geneR[index_complexR]
        sorted_indexCR=[]
        complexLnewR=[]
        for i in complexR.tolist():
            if i in complex_inputname.tolist():
                sorted_indexCR.append(complex_inputname.tolist().index(i))
                complexLnewR=complexLnewR+[j for j,v in enumerate(geneR.tolist()) if v==i]
        complexLnewR = list(set(complexLnewR))
        complexLnewR.sort()
        data_complex=computeExpr_complex(complex_input, data_use, sorted_indexCR,data_rownames)

        dataR[complexLnewR,] = data_complex

    p_idx=[]
    for i in Pair_name.tolist():
        if i in pairname.tolist():
            p_idx.append(pairname.tolist().index(i))
    maxlen=np.max(np.array(len_idx))
    Feature = np.zeros((len(Cell1),dataL.shape[1]*2))
    for i in range(len(Cell1)):
        L_feature=np.zeros((1,dataL.shape[1]))
        R_feature=np.zeros((1,dataR.shape[1]))
        L_feature[0,idx[Cell1[i]]] = dataL[p_idx[i],idx[Cell1[i]]]
        R_feature[0,idx[Cell2[i]]] = dataR[p_idx[i],idx[Cell2[i]]]
        Feature[i,:]=np.hstack((L_feature,R_feature))
    Feature_sparse = sparse.csc_matrix(Feature)
    svd = TruncatedSVD(n_components=1600, n_iter=10, random_state=42)
    Feature1 = svd.fit_transform(Feature_sparse)
    Feature1 = Feature1.astype(str)
    Feature_idx=np.arange(0,len(Cell1))
    Featureall=np.column_stack((Feature_idx,Feature1))
    np.savetxt("./output/Feature.csv",Featureall,fmt="%s",delimiter=",")


def graph():
    complex_inputdata = './output/complex_input.csv'
    with open(complex_inputdata,encoding = 'gbk') as f:

        complex_inputname=np.loadtxt(complex_inputdata,str,delimiter = ",",skiprows=1)[:,0]
        #print(complex_inputname)
        for i in range(len(complex_inputname)):
            complex_inputname[i]=ast.literal_eval(complex_inputname[i])
        subunit1 = np.loadtxt(complex_inputdata,str,delimiter = ",",skiprows=1)[:,1]
        for i in range(len(subunit1)):
            subunit1[i]=ast.literal_eval(subunit1[i])
        subunit2 = np.loadtxt(complex_inputdata,str,delimiter = ",",skiprows=1)[:,2]
        for i in range(len(subunit2)):
            subunit2[i]=ast.literal_eval(subunit2[i])
        subunit3 = np.loadtxt(complex_inputdata,str,delimiter = ",",skiprows=1)[:,3]
        for i in range(len(subunit3)):
            subunit3[i]=ast.literal_eval(subunit3[i])
        subunit4 = np.loadtxt(complex_inputdata,str,delimiter = ",",skiprows=1)[:,4]
        for i in range(len(subunit4)):
            subunit4[i]=ast.literal_eval(subunit4[i])


    complex_input1=np.vstack((np.array(subunit1),np.array(subunit2)))
    complex_input2=np.vstack((np.array(subunit3),np.array(subunit4)))
    complex_input=np.vstack((complex_input1,complex_input2)).T


    pairLRsigdata = './output/pair_use_top.csv'
    geneL = pd.read_csv(pairLRsigdata).values[:,3]
    geneR = pd.read_csv(pairLRsigdata).values[:,4]
    pairname=pd.read_csv(pairLRsigdata).values[:,1]
    nLR=len(geneL)
    data_project='./output/data_project.csv'
    with open(data_project,encoding = 'gbk') as f:
        data = np.loadtxt(data_project,str,delimiter = ",",skiprows=1)[:,1:]
        data_rownames=np.loadtxt(data_project,str,delimiter = ",",skiprows=1)[:,0]
        data=(data.astype(np.float64))
        for i in range(len(data_rownames)):
            data_rownames[i]=ast.literal_eval(data_rownames[i])

    metadata=np.loadtxt('./output/meta.csv',str,delimiter = ",",skiprows=1)[:,-1]
    for i in range(len(metadata)):
        metadata[i]=ast.literal_eval(metadata[i]).replace (' ','')
    #np.savetxt("./output/labelname.csv",metadata,fmt="%s",delimiter=",")
    labelname = np.unique(metadata)
    idx = defaultdict(list)

    for j in range(len(labelname)):
        for i in range(len(metadata)):
            if labelname[j]== metadata[i]:
                idx[labelname[j]].append((i))
    len_idx=[]
    for i in range(len(labelname)):
        len_idx.append(len(idx[labelname[i]]))

    #net=np.loadtxt('./output/df_net.csv',str,delimiter = ",",skiprows=1)
    net=pd.read_csv('./output/df_net.csv').values
    Cell_s=net[:,1]
    Cell_t=net[:,2]
    PairL=net[:,3]
    PairR=net[:,4]
    P_label=net[:,6]
    Pair_name=net[:,7]
    Cell1 = []
    Cell2 = []
    for i in range(len(Cell_s)):
        Cell1.append(Cell_s[i].replace (' ',''))
        Cell2.append(Cell_t[i].replace (' ',''))

    pair_name_label=[]
    for i in range(len(Cell1)):
        pair_name_label.append(Cell1[i]+'_'+Cell2[i]+'_'+Pair_name[i])


    data_use = data/np.max(data)
    nC=np.shape(data_use)[1]
    data_rownames=data_rownames.tolist()
    sorted_indexL = []
    for i in geneL.tolist():
        if i in data_rownames:
            sorted_indexL.append(data_rownames.index(i))
    sorted_indexR = []
    for i in geneR.tolist():
        if i in data_rownames:
            sorted_indexR.append(data_rownames.index(i))

    data_rownames=np.array(data_rownames)
    index_singleL=np.where(np.isin(geneL,data_rownames,invert=False)==True)
    index_complexL=np.where(np.isin(geneL,data_rownames,invert=False)==False)
    index_singleR=np.where(np.isin(geneR,data_rownames,invert=False)==True)
    index_complexR=np.where(np.isin(geneR,data_rownames,invert=False)==False)
    dataL1 = data_use[sorted_indexL,]


    dataL=np.zeros((nLR,nC))
    dataL[index_singleL,] = dataL1
    if len(index_complexL[0]) > 0:
        complexL = geneL[index_complexL]
        sorted_indexCL=[]
        complexLnewL=[]

        for i in complexL.tolist():
            if i in complex_inputname.tolist():
                sorted_indexCL.append(complex_inputname.tolist().index(i))
                complexLnewL=complexLnewL+[j for j,v in enumerate(geneL.tolist()) if v==i]
        
        complexLnewL = list(set(complexLnewL))
        complexLnewL.sort()
        data_complex=computeExpr_complex(complex_input, data_use, sorted_indexCL,data_rownames)
        dataL[complexLnewL,] = data_complex


    dataR1 = data_use[sorted_indexR,]
    dataR=np.zeros((nLR,nC))
    dataR[index_singleR,] = dataR1

    if len(index_complexR[0]) > 0:
        complexR = geneR[index_complexR]
        sorted_indexCR=[]
        complexLnewR=[]
        for i in complexR.tolist():
            if i in complex_inputname.tolist():
                sorted_indexCR.append(complex_inputname.tolist().index(i))
                complexLnewR=complexLnewR+[j for j,v in enumerate(geneR.tolist()) if v==i]
        complexLnewR = list(set(complexLnewR))
        complexLnewR.sort()
        data_complex=computeExpr_complex(complex_input, data_use, sorted_indexCR,data_rownames)

        dataR[complexLnewR,] = data_complex

    Pair_LR = np.hstack((dataL,dataR))
    Pair_LR_sparse = sparse.csc_matrix(Pair_LR)
    svd = TruncatedSVD(n_components=500, n_iter=10, random_state=42)
    #sparse.save_npz("../Data/pathway_pre/"+str(pathwayname[m])+".npz",pathway_sparse)

    Pair_LR1 = svd.fit_transform(Pair_LR_sparse)
    Pair_LR = np.column_stack((pairname,Pair_LR1))

    labels=Pair_LR[:,0]
    Pair_LRdata = Pair_LR[:,1:]
    data=Pair_LRdata.astype(np.float32)

    AJ=np.corrcoef((data))

    AJ[AJ>=0.95]=1
    AJ[AJ<0.95]=0


    row, col = np.diag_indices_from(AJ)
    AJ[row,col] = 0
    A=np.array(np.where(AJ>0))
    #print(A.T)
    np.savetxt("./output/graph/test.cites",A.T,fmt="%s",delimiter=" ")

    index=np.arange(0,(data.shape[0])).astype(str)
    data=data.astype(str)
    data1=np.insert(data, 0, values=index, axis=1)
    data2=np.column_stack((data1,labels))
    #print(data2.shape)
    np.savetxt("./output/graph/test.content",data2,fmt="%s",delimiter=" ")
    adj, features = load_data(path="./output/graph/", dataset="test")
    torch.save(adj, "./output/adj.pth")
    torch.save(features, "./output/features.pth")




if __name__ == "__main__":
    if not os.path.exists("./output/"):
        os.system('mkdir ./output/')
    if not os.path.exists("./output/graph"):
        os.system('mkdir ./output/graph')

    parser = argparse.ArgumentParser(
    description='Cell_Interaction',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--species', type=str, default='Human')
    parser.add_argument('--label_mode', default=True)
    args = parser.parse_args()
    if args.species == 'Human':
        LR_DB='./LRDB/LRDB.human.rda'
    elif args.species == 'Mouse':
        LR_DB='./LRDB/LRDB.mouse.rda'

    if args.label_mode:
        os.system('Rscript Feature.R ./input/test.rds ./input/test_cell_label.csv '+str(LR_DB))
    else:
        print("Cell Clustering...")
        os.system('python ./cluster/Cluster.py --pretain True --pretrain_epoch 50 --device cuda --Auto False')
        os.system('Rscript Feature.R ./input/test.rds ./cluster/output/cell_annotatetype.csv '+str(LR_DB))
    #print("cut graph")
    cut_graph()
    #print("Feature")
    Feature()
    #print("graph")
    graph()

