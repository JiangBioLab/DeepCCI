import argparse
import os
import warnings
import math
from collections import defaultdict
import numpy as np
import torch
from sklearn import preprocessing
import torchvision.transforms as transforms
from modelv2 import MultiOutputModel
from sklearn.metrics import jaccard_score, confusion_matrix, multilabel_confusion_matrix,hamming_loss,accuracy_score,average_precision_score,roc_auc_score,label_ranking_average_precision_score,recall_score,f1_score,precision_score
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
def checkpoint_load(model, name):
	print('Restoring checkpoint: {}'.format(name))
	model.load_state_dict(torch.load(name, map_location='cpu'))
	epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
	return epoch


def validate(model, dataloader, batch_size, adj, features, iteration, device, checkpoint=None):
	'''
	pretrained_dict = torch.load(checkpoint)
	model_dict = model.state_dict()
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'gc1' not in k)}
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'gc2' not in k)}
	model_dict.update(pretrained_dict)
	
	model.load_state_dict(model_dict)
	'''
	if checkpoint is not None:
		checkpoint_load(model, checkpoint)
	#adj, features = load_data()
	
	model.eval()
	#batch_size=512
	results = defaultdict(list)
	with torch.no_grad():
		avg_loss2 = 0
		F1=0
		Recall=0
		ACC=0
		Precision=0
		#accuracy_mutil6=0
		Target=[]
		Predict=[]
		for batch in dataloader:
			data = batch['data']
			data=torch.reshape(data,(batch_size,1,int(math.sqrt(data.size(1))),int(math.sqrt(data.size(1))))).to(torch.float32)

			target_labels = batch['labels']
			target_labels = {t: target_labels[t].to(device) for t in target_labels}
			output = model(data.to(device),features.to(device), adj.to(device))
			val_loss = model.get_loss(output, target_labels)
			#n_classes=output['class'].shape[1]
			avg_loss2 += val_loss.item()

			batch_F1,batch_Recall,batch_ACC,batch_Precision = \
			calculate_metrics(output, target_labels)
			F1 += batch_F1
			Recall += batch_Recall
			ACC += batch_ACC
			Precision += batch_Precision
			Target += (target_labels['labels'].cpu().tolist())
			Predict += (output['sigmoid'].cpu().detach().numpy().tolist())
	n_samples = len(dataloader)
	avg_loss2 /= n_samples
	F1 /=n_samples
	Recall /=n_samples
	ACC /=n_samples
	Precision /=n_samples

	print('-' * 72)
	print("Validation  loss: {:.4f}, F1: {:.4f}, Recall: {:.4f}, ACC: {:.4f},Precision: {:.4f},".format(avg_loss2, F1,Recall,ACC,Precision))

	model.train()
	

	return avg_loss2,F1,Recall,ACC,Precision,Target,Predict





def calculate_metrics(output, target):
	predicted_class_labels = output['sigmoid'].cpu()

	predicted_mutil_labels = output['class'].cpu()

	gt_mutil_labels = target['labels'].cpu()
	with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
		warnings.simplefilter("ignore")

		y_true=gt_mutil_labels.numpy()

		#print(y_true)
		y_score =predicted_mutil_labels.detach().numpy()
		y_pred = predicted_class_labels.detach().numpy()

		F1 = f1_score(y_true,(y_pred > 0.5).astype(float))
		Recall = recall_score(y_true, (y_pred > 0.5).astype(float))
		Acc = accuracy_score(y_true, (y_pred > 0.5).astype(float))
		Precision = precision_score(y_true,(y_pred > 0.5).astype(float))

	return F1,Recall,Acc,Precision


