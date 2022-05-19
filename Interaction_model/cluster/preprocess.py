import numpy as np
import pandas as pd
import ast
import operator
from itertools import chain
import math
import os
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from scipy.spatial import distance_matrix, minkowski_distance, distance
import scipy.sparse
import sys
import pickle
import csv
import networkx as nx
import numpy as np
from sklearn.ensemble import IsolationForest
import time
from multiprocessing import Pool
import multiprocessing 
from igraph import *
from sklearn import preprocessing

def calculateKNNgraphDistanceMatrixStatsSingleThread(featureMatrix, distanceType='euclidean', k=10, param=None):

	edgeList=[]

	p_time = time.time()
	for i in np.arange(featureMatrix.shape[0]):
		tmp=featureMatrix[i,:].reshape(1,-1)
		distMat = distance.cdist(tmp,featureMatrix, distanceType)
		res = distMat.argsort()[:k+1]
		tmpdist = distMat[0,res[0][1:k+1]]

		boundary = np.mean(tmpdist)+np.std(tmpdist)
		for j in np.arange(1,k+1):
			# TODO: check, only exclude large outliners
			# if (distMat[0,res[0][j]]<=mean+std) and (distMat[0,res[0][j]]>=mean-std):
			
			if distMat[0,res[0][j]]<=boundary:
				weight = 1.0
			else:
				weight = 0.0
			
			#weight = 1.0
			edgeList.append((i,res[0][j],weight))

	return edgeList

def calculateKNNgraphDistanceMatrix(featureMatrix, distanceType='euclidean', k=10):

	distMat = distance.cdist(featureMatrix,featureMatrix, distanceType)
		
	edgeList=[]

	for i in np.arange(distMat.shape[0]):
		res = distMat[:,i].argsort()[:k]
		for j in np.arange(k):
			edgeList.append((i,res[j]))
	
	return edgeList

def edgeList2edgeDict(edgeList, nodesize):
	graphdict={}
	tdict={}

	for edge in edgeList:
		end1 = edge[0]
		end2 = edge[1]
		tdict[end1]=""
		tdict[end2]=""
		if end1 in graphdict:
			tmplist = graphdict[end1]
		else:
			tmplist = []
		tmplist.append(end2)
		graphdict[end1]= tmplist

	#check and get full matrix
	for i in range(nodesize):
		if i not in tdict:
			graphdict[i]=[]

	return graphdict

def generateAdj(featureMatrix, graphType='KNNgraph', para = None):
	"""
	Generating edgeList 
	"""
	edgeList = None
	adj = None
	#edgeList = calculateKNNgraphDistanceMatrix(featureMatrix)
	edgeList = calculateKNNgraphDistanceMatrixStatsSingleThread(featureMatrix)
	graphdict = edgeList2edgeDict(edgeList, featureMatrix.shape[0])
	adj = nx.adjacency_matrix(nx.from_dict_of_lists(graphdict))
	return adj, edgeList


def generateLouvainCluster(edgeList):
	"""
	Louvain Clustering using igraph
	"""
	Gtmp = nx.Graph()
	Gtmp.add_weighted_edges_from(edgeList)
	W = nx.adjacency_matrix(Gtmp)
	W = W.todense()
	graph = Graph.Weighted_Adjacency(
		W.tolist(), mode=ADJ_UNDIRECTED, attr="weight", loops=False)
	louvain_partition = graph.community_multilevel(
		weights=graph.es['weight'], return_levels=False)
	size = len(louvain_partition)
	hdict = {}
	count = 0
	for i in range(size):
		tlist = louvain_partition[i]
		for j in range(len(tlist)):
			hdict[tlist[j]] = i
			count += 1

	listResult = []
	for i in range(count):
		listResult.append(hdict[i])

	return listResult, size

def read_data(filename, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index = utilss.decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index = utilss.decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns


def prepro(data_type,filename):
    if data_type == 'csv':
        data_path = "./dataset/" + filename + "/data.csv"
        label_path = "./dataset/" + filename + "/label.csv"
        X = pd.read_csv(data_path, header=0, index_col=0, sep=',')
        #X = np.expm1(X)
        cell_label = pd.read_csv(label_path).values[:,-1]
    if data_type == 'txt':
        data_path = "./dataset/" + filename + "/data.txt"
        label_path = "./dataset/" + filename + "/label.csv"
        X = pd.read_csv(data_path, header=0, index_col=0, sep=',')
        cell_label = pd.read_csv(label_path).values[:,-1]
    if data_type == 'h5':
        data_path = "./dataset/" + filename + "/data.h5"
        mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)
        if isinstance(mat, np.ndarray):
            X = np.array(mat)
        else:
            X = np.array(mat.toarray())
        cell_name = np.array(obs["cell_type1"])
        cell_type, cell_label = np.unique(cell_name, return_inverse=True)
    return X, cell_label


def Selecting_highly_variable_genes(X, highly_genes):
    adata = sc.AnnData(X)
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    sc.pp.scale(adata)
    data = adata.X

    return data

def normalize(adata, copy=True, highly_genes = None, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6: # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata

def getcluster(edgeList):

	#featureMatrix = pd.read_csv('./output/Top2000.csv').values[:,1:].T

	
	listResult, size = generateLouvainCluster(edgeList)
	n_clusters = len(np.unique(listResult))
	#print('Louvain cluster: '+str(n_clusters))
	return n_clusters
