import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, Adamax
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.metrics import silhouette_score, davies_bouldin_score 
from sklearn.cluster import SpectralBiclustering,KMeans, kmeans_plusplus, DBSCAN,SpectralClustering
from evaluation import eva_pretrain
import umap
import argparse

#torch.cuda.set_device(3)


class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
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

        return x_bar, z


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pretrain_ae(model,dataset,m,device,n_clusters,epoch,Auto=True):
    train_loader = DataLoader(dataset, batch_size=None, shuffle=True)
    #device = args.device
    #print(device)
    optimizer = Adamax(model.parameters(), lr=1e-2)
    for epoch in range(epoch):
        adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            x_bar, _ = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).to(device).float()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
        
            #for i in range(0,100,10):
            if z.shape[0] < 5000:
                resolution = 0.8
            else:
                resolution = 0.5
            if Auto:
                n_clusters = int(clusters*resolution) if int(clusters*resolution)>=3 else 2
            #print(n_clusters)
            kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(z.data.cpu().numpy())
            silhouette =eva_pretrain(z.data.cpu().numpy(), kmeans.labels_, epoch)
            #print(kmeans.labels_)
            #print('epoch{} loss: {}'.format(epoch, loss),'silhouette {:.4f}'.format(silhouette))   
            torch.save(model.state_dict(), './output/model/test'+str(m)+'.pkl')

    return silhouette


