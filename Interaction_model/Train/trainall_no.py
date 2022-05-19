import argparse
import os
from datetime import datetime
import pandas as pd
import torch
import math
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn import preprocessing
from sklearn.model_selection import KFold
from collections import defaultdict
#from utils import *
from scipy import sparse
import torchvision.transforms as transforms
from dataset import ForDataset, AttributesDataset
from modelv2 import MultiOutputModel
from test import calculate_metrics, validate
from torch.utils.data import DataLoader
import numpy as np
import os
from torch.nn import init
import time 
#import torchcontrib
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold,StratifiedKFold
cpu_num = 6
from scipy.stats import ks_2samp
torch.set_num_threads(cpu_num)
def sum_dict(a,b):
	temp = dict()
	for key in a.keys()| b.keys():
		temp[key] = sum([d.get(key, 0) for d in (a, b)])
	return temp

def get_cur_time():
	return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def checkpoint_save(model, name, epoch):
	f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
	torch.save(model.state_dict(), f)
	print('Saved checkpoint:', f)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Training pipeline')
	parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
	parser.add_argument('--seed', type=int, default=72, help='Random seed.')
	#parser.add_argument('--attributes_file', type=str, default='/home/user/yangwenyi/dataNew/dataall/pathwaylabels.csv',help="Path to the file with attributes")
	parser.add_argument('--device', type=str, default='cuda', help="Device: 'cuda' or 'cpu'")
	parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,metavar='LR', help='initial learning rate')
	parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.01, type=float, metavar='LR', help='learning rate for pre-trained layers')
	parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
	parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
	parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
	args = parser.parse_args()
	

	start_epoch = 1
	N_epochs = args.epochs
	batch_size = args.batch_size
	#print(batch_size)
	num_workers = 0  # number of processes to handle dataset loading
	device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
	print(device)
	#adj, features = load_data()
	adj = torch.load("./data/adj.pth")
	features = torch.load("./data/features.pth")
	print(args)
	#print(features.shape)
	Feature=pd.read_csv('./data/Feature_data.csv',header=None).values
	data_train=Feature[:,1:-1].astype(np.float32)
	label=Feature[:,[0,-1]].astype(np.float32).astype(np.int32)

	MinMax = preprocessing.MinMaxScaler()
	data_train=MinMax.fit_transform(data_train)
	data_train=preprocessing.normalize(data_train, norm='l2')

	train_dataset = ForDataset(data_train, label)
	train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)


	model = MultiOutputModel(nfeat=features.shape[1],nlabel=features.shape[0]).to(device)

	optimizer = torch.optim.Adamax(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
	scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=False, threshold=0.000001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
	savedir="./model/"
	#os.makedirs(logdir, exist_ok=True)
	os.makedirs(savedir, exist_ok=True)

	n_train_samples = len(train_dataloader)

	print("Starting training ...")
	for epoch in range(start_epoch, N_epochs + 1):
		#model.train()
		batch_size = args.batch_size
		#print("epoch:",epoch)
		t=time.time()

		total_loss2 = 0
		F1=0
		Recall=0
		ACC=0
		Precision=0
		for batch in train_dataloader:
			optimizer.zero_grad()
			data = batch['data']
			data=torch.reshape(data,(args.batch_size,1,int(math.sqrt(data.size(1))),int(math.sqrt(data.size(1))))).to(torch.float32)
			target_labels = batch['labels']
			target_labels = {t: target_labels[t].to(device) for t in target_labels}
			#print(target_labels)
			output = model(data.to(device),features.to(device), adj.to(device))
			loss = model.get_loss(output, target_labels)
			loss2= loss.item()
			total_loss2 += loss2
			batch_F1,batch_Recall,batch_ACC,batch_Precision = \
			calculate_metrics(output, target_labels)
			F1 += batch_F1
			Recall += batch_Recall
			ACC += batch_ACC
			Precision += batch_Precision
			loss.backward()
			optimizer.step()

		#optimizer.swap_swa_sgd()
		print("epoch {:4d}, loss: {:.4f},F1: {:.4f}, recall: {:.4f}, ACC: {:.4f}, Precision: {:.4f},time: {:.4f}".format(
			epoch,
			total_loss2 / n_train_samples,
			F1 / n_train_samples,
			Recall / n_train_samples,
			ACC / n_train_samples,
			Precision / n_train_samples,
			time.time()-t
				))

		if epoch % 5 == 0: 
			checkpoint_save(model, savedir, epoch)

