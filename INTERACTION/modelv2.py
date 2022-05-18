import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
#import torchvision.models as models
from Mobilev2 import MobileNetV2

class FocalLoss(nn.Module):
	def __init__(self, alpha=0.65, gamma=2, logits=False, reduce=True):
		super(FocalLoss, self).__init__()
		self.alpha = alpha
		self.gamma = gamma
		self.logits = logits
		self.reduce = reduce

	def forward(self, inputs, targets):
		if self.logits:
			BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
		else:
			BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
		pt = torch.exp(-BCE_loss)
		F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

		if self.reduce:
			return torch.mean(F_loss)
		else:
			return F_loss



class GraphConvolution(nn.Module):
	"""
	Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
	"""

	def __init__(self, in_features, out_features, bias=False):
		super(GraphConvolution, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weight = Parameter(torch.Tensor(in_features, out_features))
		if bias:
			self.bias = Parameter(torch.Tensor(1, 1, out_features))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-stdv, stdv)
		if self.bias is not None:
			self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input, adj):
		support = torch.matmul(input, self.weight)
		output = torch.matmul(adj, support)
		if self.bias is not None:
			return output + self.bias
		else:
			return output

	def __repr__(self):
		return self.__class__.__name__ + ' (' \
			+ str(self.in_features) + ' -> ' \
			+ str(self.out_features) + ')'
class MultiOutputModel(nn.Module):

	def __init__(self,nfeat,nlabel):

		super().__init__()
		models=MobileNetV2()
		self.base_model = MobileNetV2().features  # take the model without classifier
		last_channel = MobileNetV2().last_channel  # size of the layer before classifier
		self.pool = nn.AdaptiveMaxPool2d((1, 1))
		self.gc1 = GraphConvolution(nfeat, 512)
		self.gc2 = GraphConvolution(512, last_channel)
		self.relu = nn.LeakyReLU(0.2)
		self.fc = nn.Linear(nlabel,1)
		self.dropout=nn.Dropout(0.2)

	def forward(self, x, feature, adj):
		x = self.base_model(x)
		x = self.pool(x)
		x = self.dropout(x)
		# reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
		x = torch.flatten(x, 1)
		feature = self.gc1(feature, adj)
		#feature = self.dropout(feature)
		feature = self.relu(feature)
		feature = self.gc2(feature, adj)
		#feature = self.dropout(feature)
		feature = self.relu(feature)
		feature = feature.transpose(0, 1)

		x = torch.matmul(x,feature)
		x = self.fc(x)
		x = self.dropout(x)
		x = x.squeeze(-1)
		xt = torch.sigmoid(x)
		#print(y)
		return {'class': x,'sigmoid': xt
			
			}
	def initialize(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight, gain=1)
				#print(m.weight)

	def get_config_optim(self, lr, lrp):
		return [
				{'params': self.base_model.parameters(), 'lr': lr },
				{'params': self.gc1.parameters(), 'lr': lrp},
				{'params': self.gc2.parameters(), 'lr': lrp},
				]
	def get_loss(self, net_output, ground_truth):
			#crition=nn.BCEWithLogitsLoss()
			crition=FocalLoss()
			loss = crition(net_output['sigmoid'].float(), ground_truth['labels'].float())
			#loss = crition(net_output['class'].float(), ground_truth['labels'].float())
			#crition2 = nn.MultiLabelSoftMarginLoss()
			#loss = crition2(net_output['mutil'], ground_truth['multi_labels'])
			return loss