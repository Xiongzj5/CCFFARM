import torch
import random
import numpy as np
import pandas as pd

class network:
	dataset = []
	epochs = 100
	attr = 0

	def __init__(self, attr_size, h1_size, h2_size, out_size, ep, learning_rate):
		self.net = torch.nn.Sequential(
			torch.nn.Linear(attr_size, h1_size), 
			torch.nn.ReLU(),
			# torch.nn.Linear(h1_size, h2_size), 
			# torch.nn.Sigmoid(),   
			# torch.nn.Linear(h2_size ,out_size), 
			torch.nn.Linear(h1_size ,out_size), 
		)
		self.attr = attr_size
		self.epochs = ep
		self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate)
		self.loss_func = torch.nn.MSELoss()
	
	def read_data(self, path):
		self.dataset = pd.read_csv(path, header=None, sep=',')

	def accuracy_measure(self, rule, ann_weight): 
		# 权重赋值
		base = 0
		for layer in self.net:
			if isinstance(layer, torch.nn.Linear):
				in_d = len(layer.weight.data)
				out_d = len(layer.weight.data[0])
				for i in range(in_d):
					layer.weight.data[i] = torch.from_numpy(ann_weight[base + i * out_d : base + (i + 1) * out_d])
				base += in_d * out_d

		# rule = [11, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
		# print('rule', rule)

		# Get Training Dataset
		data_size = len(self.dataset)
		training_size = int(data_size * 0.8)
		training_set = np.zeros((training_size, self.attr))
		for i in range(training_size):
			for j in range(self.attr):
				training_set[i][j] = self.dataset.iloc[i, j] * rule[j + 1]
		x = torch.from_numpy(training_set).float()

		rhs = rule[0] - 1   # x11 - x20
		y = np.array(self.dataset.iloc[0:training_size, rhs]).reshape((training_size, 1))
		y = torch.from_numpy(y).float()

		# print(x[0:5])
		# print(y[0:5])

		# BP training
		for j in range(self.epochs):
			prediction = self.net(x)        # input x and predict based on x
			loss = self.loss_func(prediction, y) 
			self.optimizer.zero_grad()      # clear gradients for next train
			loss.backward()                 # backpropagation, compute gradients
			self.optimizer.step()           # apply gradients	
		
		# Get Test Dataset
		test_size = data_size - training_size
		test_set = np.zeros((test_size, self.attr))
		for i in range(test_size):
			for j in range(self.attr):
				test_set[i][j] = self.dataset.iloc[training_size + i, j] * rule[j + 1]
		test_x = torch.from_numpy(test_set).float()

		test_y = np.array(self.dataset.iloc[training_size:, rhs]).reshape((test_size, 1))
		test_y = torch.from_numpy(test_y).float()

		# Testing
		pred_y = self.net(test_x)
		loss = self.loss_func(pred_y, test_y)

		# print(test_y[0:5])
		# print(pred_y[0:5])
		# print('accuracy = ', 1 - loss)
						
		return 1 - loss

