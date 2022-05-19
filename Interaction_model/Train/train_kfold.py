import argparse
import os
from datetime import datetime
import pandas as pd
import torch
import math
import warnings
warnings.filterwarnings("ignore")
import math
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
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
#from modelv2 import MultiOutputModel
from test import calculate_metrics, validate
from torch.utils.data import DataLoader
import numpy as np
import os
from torch.nn import init
import time 
#import torchcontrib
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold,StratifiedKFold
cpu_num = 2
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
def l2_penalty(w):
	return (w**2).sum() / 2



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Training pipeline')
	parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
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
	num_workers = 2  # number of processes to handle dataset loading
	#device = torch.device("cpu")
	
	device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
	#adj, features = load_data(path="./Predict/human/graph/", dataset="test")
	
	adj = torch.load("./data/adj.pth")
	features = torch.load("./data/features.pth")
	print(args)
	#print(features.shape)
	Feature=pd.read_csv('./data/Feature_data.csv',header=None).values
	#Feature=np.loadtxt('./Data/Feature_data.csv',str,delimiter = ",",skiprows=0)
	data_train=Feature[:,1:-1].astype(np.float32)
	label=Feature[:,[0,-1]].astype(np.float32).astype(np.int32)
	skf = StratifiedKFold(n_splits=5, shuffle=True)
	train_F1=[];train_Recall=[];train_Hamming=[];train_Precision=[];train_AP=[]
	test_F1=[];test_Recall=[];test_Hamming=[];test_Precision=[];test_AP=[]
	kn=0
	plt.figure()
	mean_tpr = 0.0
	MinMax = preprocessing.MinMaxScaler()
	data_train=MinMax.fit_transform(data_train)
	data_train=preprocessing.normalize(data_train, norm='l2')
	mean_fpr = np.linspace(0, 1, 100)
	for train_index, test_index in skf.split(data_train,label[:,-1]):
		print(str(kn+1)+"fold:")
		results = defaultdict(list)
		#print(len(train_index))
		#data_train = preprocessing.scale(data_train)

		'''
		Scaler=preprocessing.StandardScaler()
		data_train = Scaler.fit_transform(data_train)
		MinMax = preprocessing.MinMaxScaler()
		data_train=MinMax.fit_transform(data_train)

		data_train=preprocessing.normalize(data_train, norm='l2')
		'''
		train_X, train_y = data_train[train_index], label[train_index]
		test_X, test_y = data_train[test_index], label[test_index]
		
		train_X = torch.from_numpy(train_X)
		test_X = torch.from_numpy(test_X)

		#test_X = Scaler.fit_transform(test_X)
		#test_X=preprocessing.normalize(test_X, norm='l1')
		train_dataset = ForDataset(train_X, train_y)
		train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,drop_last=True)
		val_dataset = ForDataset(test_X,test_y)
		val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers,drop_last=True)
		model = MultiOutputModel(nfeat=features.shape[1],nlabel=features.shape[0]).to(device)
		#print(model)
		#optimizer = torch.optim.Adam(model.get_config_optim(args.lr, args.lrp),weight_decay=args.weight_decay)
		#optimizer = torch.optim.Adamax(model.get_config_optim(args.lr, args.lrp),weight_decay=args.weight_decay)
		optimizer = torch.optim.Adamax(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
		#optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

		#optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
		#optimizer = torch.optim.RMSprop(model.parameters(),lr=args.lr,alpha=0.99, eps=1e-08,momentum=args.momentum,weight_decay=args.weight_decay)

		#scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
		#optimizer = torchcontrib.optim.SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.01)	
		#print('SGD')
		#optimizer = torch.optim.Adadelta(model.get_config_optim(args.lr, args.lrp), lr=args.lr, rho=0.9, eps=1e-6, weight_decay=args.weight_decay)

		#optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),momentum=args.momentum,weight_decay=args.weight_decay)
		scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=False, threshold=0.0000001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

		#logdir = os.path.join('./logs/', get_cur_time())
		#savedir = os.path.join('./checkpoint/', get_cur_time())
		#savedir="./model/panc/train_kfold/"
		#os.makedirs(logdir, exist_ok=True)
		resultdir = "./results/"
		os.makedirs(resultdir, exist_ok=True)
		#logger = SummaryWriter(logdir)

		n_train_samples = len(train_dataloader)

		print("Starting training ...")
		loss_early = 10000
		for epoch in range(start_epoch, N_epochs + 1):
			
			model.train()
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
				'''
				re_l1=0
				for param in model.parameters():
					re_l1+=torch.sum(torch.abs(param))
				'''
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
			#print(accuracy_mutil6/n_train_samples)
			results['train_loss'].append(total_loss2 / n_train_samples)
			results['F1_train'].append(F1 / n_train_samples)
			results['recall_train'].append(Recall / n_train_samples)
			results['Pre_train'].append(Precision / n_train_samples)
			#logger.add_scalar('train_loss', total_loss2 / n_train_samples, epoch)
			
			#if epoch % 5 == 0: 
			#	checkpoint_save(model, savedir, epoch)

			if epoch % 5 == 0:
				#checkpoint = os.path.join(savedir, 'checkpoint-{:06d}.pth'.format(epoch))
				val_loss2,F1_val,Recall_val,Hamming_val,Precision_val,Target,Predict = validate(model, val_dataloader, batch_size ,adj, features, epoch, device,checkpoint=None)
				results['val_loss'].append(val_loss2)
				results['F1_val'].append(F1_val)
				results['recall_val'].append(Recall_val)
				results['presion_val'].append(Precision_val)
				#print(Target)
				#print(Predict)
				test_F1.append(F1_val);test_Recall.append(Recall_val);test_Hamming.append(Hamming_val)
				test_Precision.append(Precision_val)
				fpr, tpr, thresholds = roc_curve(Target, Predict)
				mean_tpr += np.interp(mean_fpr, fpr, tpr)
				mean_tpr[0] = 0.0
				roc_auc = auc(fpr, tpr)
				plt.plot(fpr, tpr, lw=1, label='Fold {0:.0f} (AUC = {1:.2f})'.format(kn+1, roc_auc))
				'''
				if loss_early > val_loss2:
					if epoch < N_epochs:
						#print("early stopping")
						loss_early = val_loss2
					elif epoch == N_epochs:
						test_F1.append(F1_val);test_Recall.append(Recall_val);test_Hamming.append(Hamming_val)
						test_Precision.append(Precision_val)
						fpr, tpr, thresholds = roc_curve(Target, Predict)
						mean_tpr += np.interp(mean_fpr, fpr, tpr)
						mean_tpr[0] = 0.0
						roc_auc = auc(fpr, tpr)
						plt.plot(fpr, tpr, lw=1, label='Fold {0:.0f} (AUC = {1:.2f})'.format(kn+1, roc_auc))
				else:
					if epoch>30:
						print("early stopping")
						checkpoint = os.path.join(savedir, 'checkpoint-{:06d}.pth'.format(epoch-5))
						val_loss2,F1_val,Recall_val,Hamming_val,Precision_val,Target,Predict = validate(model, val_dataloader, batch_size ,adj, features, epoch, device,checkpoint=checkpoint)
					
						#if epoch % N_epochs == 0:
						#train_F1.append(F1 / n_train_samples);train_Recall.append(Recall / n_train_samples)
						#train_Hamming.append(ACC / n_train_samples);train_Precision.append(Precision / n_train_samples)
						test_F1.append(F1_val);test_Recall.append(Recall_val);test_Hamming.append(Hamming_val)
						test_Precision.append(Precision_val)
						fpr, tpr, thresholds = roc_curve(Target, Predict)
						mean_tpr += np.interp(mean_fpr, fpr, tpr)
						mean_tpr[0] = 0.0
						roc_auc = auc(fpr, tpr)
						plt.plot(fpr, tpr, lw=1, label='Fold {0:.0f} (AUC = {1:.2f})'.format(kn+1, roc_auc))
						break
					else:
						pass
				'''
		#plot_results(results, 10,"./results/results"+str(kn)+".png")
		kn+=1

	mean_tpr /= kn
	mean_tpr[-1] = 1.0
	mean_auc = auc(mean_fpr, mean_tpr)

	plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean (ROC = {0:.2f})'.format(mean_auc), lw=2)

	plt.xlim([-0.05, 1.05])
	plt.ylim([-0.05, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.tight_layout()
	plt.savefig('./results/ROC.pdf',dpi=500)
	F1_test= np.mean(np.array(test_F1));Recall_test= np.mean(np.array(test_Recall));
	ACC_test= np.mean(np.array(test_Hamming));Precision_test= np.mean(np.array(test_Precision));

	print("Test:F1: {:.4f}, Recall: {:.4f}, ACC: {:.4f}, Precision: {:.4f}".format(
				F1_test,
				Recall_test,
				ACC_test,
				Precision_test,
					))
