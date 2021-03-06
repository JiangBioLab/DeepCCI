from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
from preprocess import *
from pretrain import *
import sys
import argparse
import random
from sklearn.cluster import SpectralBiclustering,KMeans, kmeans_plusplus, DBSCAN,SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam ,SGD,Adamax
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
import umap
from evaluation import eva,eva_pretrain
from collections import Counter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import OrderedDict



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

def init_seed(opt):
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
def pretarin_cluster(n_clusters,x,device):

    print("generate cell graph...")
    Auto = args.Auto
    #calculate the number of clusters
    

    device = device

    silhouette_pre=[]
    print("Start pretrain")
    for i in range(args.pretrain_frequency):
        print("pretrain:"+str(i))
        model = AE(
            n_enc_1=100,
            n_enc_2=200,
            n_enc_3=200,
            n_dec_1=200,
            n_dec_2=200,
            n_dec_3=100,
            n_input=2000,
            n_z=5).to(device)
        dataset = LoadDataset(x)
        epoch = args.pretrain_epoch
        silhouette=pretrain_ae(model,dataset,i,device,n_clusters,epoch,Auto=Auto)
        silhouette_pre.append(silhouette)
    silhouette_pre = np.array(silhouette_pre)
    premodel_i=np.where(silhouette_pre==np.max(silhouette_pre))[0][0]
    print("Pretrain end")
    return premodel_i

class AE_train(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE_train, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class ClusterModel(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3, 
                n_input, n_z, n_clusters, v=1):
        super(ClusterModel, self).__init__()

        # autoencoder for intra information

        self.ae = AE_train(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))


        #self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        #x_bar, tra1, tra2, tra3, z = self.ae(x)
        #print(x.size())
        # GCN Module
        h1 = self.gnn_1(x, adj)
        h2 = self.gnn_2(h1, adj)
        h3 = self.gnn_3(h2, adj)
        h4 = self.gnn_4(h3, adj)
        h5 = self.gnn_5(h4, adj, active=False)
        predict = F.softmax(h5, dim=1)


        enc_h1 = F.relu(self.ae.enc_1(x))
        #print(enc_h1.size())
        enc_h2 = F.relu(self.ae.enc_2(enc_h1+h1))
        enc_h3 = F.relu(self.ae.enc_3(enc_h2+h2))
        z = self.ae.z_layer(enc_h3+h3)

        dec_h1 = F.relu(self.ae.dec_1(z+h4))
        dec_h2 = F.relu(self.ae.dec_2(dec_h1+h3))
        dec_h3 = F.relu(self.ae.dec_3(dec_h2+h2))
        x_bar = self.ae.x_bar_layer(dec_h3+h1)



        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()
def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_cluster(dataset,n_clusters,device):
    Auto=args.Auto
    if Auto:
        if z.shape[0] < 2000:
            resolution = 0.8
        else:
            resolution = 0.5
        n_clusters = int(n_clusters*resolution) if int(n_clusters*resolution)>=3 else 2
    else:
        n_clusters=n_clusters
    #device = args.device
    model = ClusterModel(100, 200, 200, 200, 200, 100,
                n_input=args.n_input,
                n_z=args.n_z,
                n_clusters=n_clusters).to(device)


    optimizer = Adamax(model.parameters(), lr=args.lr)

    # KNN Graph
    adj = load_graph(args.name)
    adj = adj.to(device)

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)
    print(n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    meta = pd.read_csv('./output/cell_name.csv',header=None).values
    index = meta[:,0]
    columns = ["labels"]
    #print(meta.shape)
    for epoch in range(args.Train_epoch):
        adjust_learning_rate(optimizer, epoch)

        if epoch % 1 == 0:
        # update_interval
            _, tmp_q, pred, _ = model(data, adj)
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
        
            res1 = tmp_q.cpu().numpy().argmax(1)       #Q
            res2 = pred.data.cpu().numpy().argmax(1)   #Z
            res3 = p.data.cpu().numpy().argmax(1)      #P
            nmi,ari,ami,silhouette=eva(tmp_q.cpu().numpy(),y, res1, str(epoch) + 'Q')
            
            print(str(epoch) + 'Q',
                        ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
                        ', ami {:.4f}'.format(ami),', silhouette {:.4f}'.format(silhouette)
                        )
            '''
            nmi,ari,ami,silhouette=eva(z.data.cpu().numpy(),y, res2, str(epoch) + 'Z')
            print(str(epoch) + 'Z',
                        ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
                        ', ami {:.4f}'.format(ami),', silhouette {:.4f}'.format(silhouette)
                        )
            nmi,ari,ami,silhouette=eva(z.data.cpu().numpy(),y, res3, str(epoch) + 'P')
            print(str(epoch) + 'P',
                        ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
                        ', ami {:.4f}'.format(ami),', silhouette {:.4f}'.format(silhouette)
                        )
            '''
        x_bar, q, pred, _ = model(data, adj)

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)
        #loss = 0.1*kl_loss + 1*ce_loss + 0.001*re_loss
        loss =0.0001*kl_loss + 0.001*ce_loss + 1*re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #np.savetxt("./output/pre_label.txt",res1,fmt="%s",delimiter=",")
    np.savetxt("./output/pre_embedding.txt",tmp_q.cpu().numpy(),fmt="%s",delimiter=",")
    pd.DataFrame(res1,index=index,columns=columns).to_csv("./output/pre_label.csv",quoting=1)
    #size = len(np.unique(res1))
    #drawUMAP(tmp_q.cpu().numpy(), res1, size, saveFlag=True)



if __name__ == "__main__":
    if not os.path.exists("./output/"):
        os.system('mkdir ./output/')
    if not os.path.exists("./output/graph"):
        os.system('mkdir ./output/graph')
    if not os.path.exists("./output/data"):
        os.system('mkdir ./output/data')
    if not os.path.exists("./output/model"):
        os.system('mkdir ./output/model')

    parser = argparse.ArgumentParser(
    description='Cell_cluster',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='cell')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed.')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--n_z', default=5, type=int)
    parser.add_argument('--pretrain_epoch', default=30, type=int)
    parser.add_argument('--pretrain_frequency', default=10, type=int)
    parser.add_argument('--Train_epoch', default=10, type=int)
    parser.add_argument('--n_input', default=2000, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    parser.add_argument('--Auto', default=False)
    parser.add_argument('--pretain', default=True)
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))

    para0=str("../input/test.rds")
    os.system("Rscript Feature.R"+" "+para0)
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
    np.savetxt("./output/graph/cell_graph.txt",np.array(edgeList)[idx,0:-1],fmt="%d")

    
    x = np.loadtxt('./output/data/cell.txt', dtype=float)
    y = np.loadtxt('../input/test_label.txt', dtype=int)
    
    if args.Auto:
        print("Auto_mode")
        auto_clusters = getcluster(edgeList)
        n_clusters = auto_clusters

        print(n_clusters)

    else:
        n_clusters = int(max(y) - min(y) + 1)

        print(n_clusters)


    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    args.Auto = False
    if args.pretain:
        premodel_i = pretarin_cluster(n_clusters,x,device)
        #print(premodel_i)
        #pretrain_path
        args.pretrain_path = './output/model/test'+str(premodel_i)+'.pkl'
    else:
        #pretain_model
        args.pretrain_path = './pretain_model/pbmc/pbmc.pkl'

    dataset = load_data(args.name)
    train_cluster(dataset,n_clusters,device)
    
    print("Cell Annotate")
    os.system("Rscript CellAnnotate.R"+" "+para0)
    Cell_type = pd.read_csv('./output/cell_type.csv').values[:,-1]
    Cell_type_new=[]
    for i in range(len(Cell_type)):
        Cell_type_new.append(Cell_type[i].split(",")[0])
    pre_cell_cluster = pd.read_csv('./output/cell_cluster.csv').values[:,-1]
    pre_label = pd.read_csv('./output/pre_label.csv').values
    Cell_label = pre_label[:,-1]
    index = pre_label[:,0]
    #print((pre_label[:,0]))
    #print(index)
    columns = ["labels"]
    #print(pre_cell_cluster)
    pre_cell_cluster = pre_cell_cluster.tolist()
    all_cell = []
    for i in range(len(Cell_label)):
        index1=pre_cell_cluster.index((Cell_label[i]))
        all_cell.append(Cell_type_new[index1].split(",")[0])

    
    pd.DataFrame(all_cell,index=index,columns=columns).to_csv("./output/cell_annotatetype.csv",quoting=1)
    featureMatrix = pd.read_csv('./output/pre_embedding.txt',header=None).values
    plotClusters(featureMatrix,Cell_label,all_cell)
