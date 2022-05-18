import csv
import pandas as pd
import numpy as np
from PIL import Image
import torch
from sklearn.model_selection import  StratifiedKFold,KFold
from scipy import sparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class AttributesDataset():
	def __init__(self, annotation_path):
		class_labels=[]
		multi_labels=[]
		with open(annotation_path) as f:
			reader = csv.DictReader(f)
			for row in reader:
				class_labels.append(row['Labels'])
			for line in reader:
				self.multi_labels.append(line[1:-2])
		self.class_labels = np.unique(class_labels)
		self.num_labels = len(self.class_labels)

		#self.class_labels_id_to_name = dict(zip(range(len(self.class_labels)), self.class_labels))
		
		#self.class_labels_name_to_id = dict(zip(self.class_labels, range(len(self.class_labels))))
		





class ForDataset(Dataset):
	def __init__(self, data_train, annotation, transform=None):
		super().__init__()

		self.transform = transform
		#self.attr = attributes
		#self.data=dd.read_csv(data_train,header=None).values.compute()
		self.data=data_train
		#print(np.shape(data))
		self.data_idx=[]
		# initialize the arrays to store the ground truth labels and paths to the images
		#self.class_labels=[]

		#self.labels=[]
		# read the annotations from the CSV file
		#with open(annotation_path) as f:
		#	reader = csv.DictReader(f)
		#	for row in reader:
		#for i in range(len())
		self.data_idx=annotation.astype(np.int32).tolist()
		#self.class_labels.append(self.attr.class_labels_name_to_id[row['Labels']])
		#self.class_labels=dd.read_csv(annotation_path).values.compute()[:,-2]

		#self.labels=annotation[:,-1]
		#print(self.multi_labels.shape)
	def __len__(self):
		return len(self.data_idx)

	def __getitem__(self, idx):
		# take the data sample by its index
		
		data=self.data
		#print(data.shape)

		#class_labels=np.array(self.class_labels).astype(np.int32)
		#labels=np.array(self.labels).astype(np.int32)
		#print(np.shape(data))
		# read image
		#img = Image.open(img_path)

		# apply the image augmentations if needed
		if self.transform:
			data = self.transform(data)
		# return the image and all the associated labels
		dict_data = {
			'data': data[idx],
		}
		#print(dict_data)

		return dict_data

