import os
import numpy as np
import pandas as pd
import umap
from collections import OrderedDict
import matplotlib.pyplot as plt


def plot(X, fig, col, size, true_labels,ann):
    ax = fig.add_subplot(1, 1, 1)
    for i, point in enumerate(X):
        ax.scatter(point[0], point[1], s=size, c=col[true_labels[i]],label=ann[i])


def plotClusters(hidden_emb, true_labels,ann):
    # Doing dimensionality reduction for plotting
    Umap = umap.UMAP(random_state=42)
    X_umap = Umap.fit_transform(hidden_emb)
    fig2 = plt.figure(figsize=(10,10),dpi=500)
    plot(X_umap, fig2, ['green','brown','purple','orange','yellow','hotpink','red','cyan','blue'], 8, true_labels,ann)
    handles, labels = fig2.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig2.legend(by_label.values(), by_label.keys(),loc="upper right")
    #fig2.legend()
    fig2.savefig("./output/UMAP.pdf")
    plt.close()


Cell_type = pd.read_csv('./output/cell_type.csv').values[:,-1]
pre_cell_cluster = pd.read_csv('./output/cell_cluster.csv').values[:,-1]
pre_label = pd.read_csv('./output/pre_label.csv').values
Cell_label = pre_label[:,-1]
index = pre_label[:,0]
#print(index)
columns = ["labels"]
#print(pre_cell_cluster)
pre_cell_cluster = pre_cell_cluster.tolist()

all_cell = []
for i in range(len(Cell_label)):
	#if (Cell_label[i]) in pre_cell_cluster:
	#print((Cell_label[i]))
	index1=pre_cell_cluster.index((Cell_label[i]))
	all_cell.append(Cell_type[index1])
pd.DataFrame(all_cell,index=index,columns=columns).to_csv("./output/cell_annotatetype.csv",quoting=1)


featureMatrix = pd.read_csv('./output/pre_embedding.txt',header=None).values
plotClusters(featureMatrix,Cell_label,all_cell)




