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



def getcluster():

	#featureMatrix = pd.read_csv('./output/Top2000.csv').values[:,1:].T

	feature = pd.read_csv('./output/Top2000.csv',header=None,low_memory=False).values
	
	Cell_name = feature[0,1:]
	featureMatrix = feature[1:,1:].T
	np.savetxt("./output/cell_name.txt",Cell_name,fmt="%s",delimiter=" ")
	data=(featureMatrix.astype(np.float32))
	np.savetxt("./output/data/cell.txt",data,fmt="%s",delimiter=" ")
	adj, edgeList = generateAdj(featureMatrix)
	#print(adj)
	idx=[]
	for i in range(np.array(edgeList).shape[0]):
	    if np.array(edgeList)[i,-1]==1.0:
	        idx.append(i)
	np.savetxt("./output/graph/cell_graph.csv",np.array(edgeList)[idx,0:-1],fmt="%d")
	listResult, size = generateLouvainCluster(edgeList)
	n_clusters = len(np.unique(listResult))
	#print('Louvain cluster: '+str(n_clusters))
	return n_clusters
