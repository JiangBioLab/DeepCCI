import argparse
import os
import warnings
import math
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from modelv2 import MultiOutputModel
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from scipy import interp
from dataset import ForDataset
from scipy import sparse
from sklearn import preprocessing
cpu_num = 3
torch.set_num_threads(cpu_num)
def checkpoint_load(model, name):
	#print('Restoring checkpoint: {}'.format(name))
	model.load_state_dict(torch.load(name, map_location='cpu'))
	epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
	return epoch


def predict(model, dataloader, batch_size, device, adj, features,checkpoint):
	#pretrained_dict = torch.load(checkpoint)
	#model_dict = model.state_dict()
	#pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'fc' not in k)}
	#pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict and 'gc2' not in k)}
	#model_dict.update(pretrained_dict)
	#model.load_state_dict(model_dict)
	checkpoint_load(model, checkpoint)
	model.eval()
	results = defaultdict(list)
	with torch.no_grad():
		Predict=[]
		for batch in dataloader:
			data = batch['data']
			data=torch.reshape(data,(batch_size,1,int(math.sqrt(data.size(1))),int(math.sqrt(data.size(1))))).to(torch.float32)

			output = model(data.to(device),features.to(device), adj.to(device))

			Predict += (output['sigmoid'].cpu().detach().numpy().tolist())

	return Predict


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Inference pipeline')
	#parser.add_argument('--checkpoint', type=str, required=True, help="Path to the checkpoint")
	parser.add_argument('--batch_size', type=int, default=None, help='batch_size')

	parser.add_argument('--device', type=str, default='cuda',
						help="Device: 'cuda' or 'cpu'")

	#parser.add_argument('--label_mode', default=True)
	args = parser.parse_args()
	num_workers = 0
	'''
	if args.label_mode:
		os.system("python Feature.py --label_mode True")
	else:
		os.system("python Feature.py --label_mode False")
	'''

	#print(batch_size)
	device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
	print(device)
	#adj, features = load_data(path="./Predict/human/graph/", dataset="test")
	Feature=pd.read_csv('./output/Feature.csv',header=None).values
	data_predict=Feature[:,1:].astype(np.float32)
	idx=Feature[:,0]
	adj = torch.load("./output/adj.pth")
	features = torch.load("./output/features.pth")
	
	#Scaler=preprocessing.StandardScaler()
	#data_predict = Scaler.fit_transform(data_predict)
	MinMax = preprocessing.MinMaxScaler()
	data_predict=MinMax.fit_transform(data_predict)
	data_predict=preprocessing.normalize(data_predict, norm='l2')
	args.batch_size = data_predict.shape[0]
	pre_dataset = ForDataset(data_predict,idx)
	pre_dataloader = DataLoader(pre_dataset, batch_size=args.batch_size, shuffle=False)
	model = MultiOutputModel(nfeat=features.shape[1],nlabel=features.shape[0]).to(device)

	#for i in range(5,100,5):
	#print(i)
	checkpoint="./model/checkpoint-000100.pth"
	Predict = predict(model, pre_dataloader, args.batch_size ,device, adj, features,checkpoint=checkpoint)
	#print(Predict)
	#label = ["label"] + Predict
	Predict=np.array(Predict)
	#print(Predict)
	Predict[Predict>0.5] = 1
	Predict[Predict<=0.5] = 0
	CC_net = pd.read_csv('./output/df_net.csv',header=None).values
	CC_net_data = CC_net[1:,1:]
	CC_pval = CC_net_data[:,5].astype(np.float16)
	CC_pval = Predict
	CC_pval=CC_pval.astype(np.int32)
	CC_net_data[:,5] = CC_pval

	index = CC_net[1:,0]

	columns = CC_net[0,1:]
	columns[5]=='label'
	pd.DataFrame(CC_net_data, index=index,columns=columns).to_csv("./output/CCI_out.csv",quoting=1)