#coding=utf-8

import torch

class SingleLSTM(torch.nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, batch_first=True):
		super(SingleLSTM, self).__init__()
		
		self.rnn = torch.nn.LSTM(     # LSTM 效果要比 nn.RNN() 好多了
			input_size=input_size,      # 图片每行的数据像素点
			hidden_size=hidden_size,     # rnn hidden unit
			num_layers=num_layers,       # 有几层 RNN layers
			dropout = dropout,
			bidirectional = False,
			batch_first=batch_first,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
		)
		
		# self.classifier = torch.nn.Linear(hidden_size, output_size) #输出层
		self.classifier = torch.nn.Sequential(torch.nn.Linear(hidden_size, hidden_size),
										torch.nn.ReLU(),
										torch.nn.Dropout(p=dropout),
										torch.nn.Linear(hidden_size, hidden_size),
										torch.nn.ReLU(),
										torch.nn.Dropout(p=dropout),
										torch.nn.Linear(hidden_size, output_size))
	
	def forward(self, x):
		n,c,h,w = x.size()
		# self.rnn.flatten_parameters()
		r_out, (h_n, h_c) = self.rnn(x.view(n,h,w), None)
		out = self.classifier(r_out[:, -1, :])
		return out

class BiLSTM(torch.nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, batch_first=True):
		super(BiLSTM, self).__init__()
		
		self.rnn = torch.nn.LSTM(
			input_size = input_size,
			hidden_size = hidden_size,
			num_layers = num_layers,
			batch_first = batch_first,
			bidirectional = True,
			dropout=dropout
		)
		
		# self.classifier = torch.nn.Linear(hidden_size*2, output_size)
		self.classifier = torch.nn.Sequential(torch.nn.Linear(hidden_size*2, hidden_size),
										torch.nn.ReLU(),
										torch.nn.Dropout(p=dropout),
										torch.nn.Linear(hidden_size, hidden_size),
										torch.nn.ReLU(),
										torch.nn.Dropout(p=dropout),
										torch.nn.Linear(hidden_size, output_size))
	
	def forward(self, x):
		n,c,h,w = x.size()
		# print(x.size())
		r_out, (h_n, h_c) = self.rnn(x.view(n,h,w), None)
		out = self.classifier(r_out[:, -1, :])
		return out

def LSTMBaseModel(model_type, input_size, hidden_size, num_layers, output_size, dropout):
	if model_type == 'singleLSTM':
		return SingleLSTM(input_size, hidden_size, num_layers, output_size, dropout)
	elif model_type == 'BiLSTM':
		return BiLSTM(input_size, hidden_size, num_layers, output_size, dropout)