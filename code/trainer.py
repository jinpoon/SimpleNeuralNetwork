from nn import Net, Optimizer
import numpy as np

class Trainer(object):
	"""docstring for Trainer"""
	def __init__(self, batchsize, dataloader, optimizer, loss_function, net):
		self.batchsize = batchsize
		self.dataloader = dataloader
		self.optimizer = optimizer
		self.loss_function = loss_function
		self.net = net

	def train(self):
		data = self.dataloader.feed()
		sum_loss = 0.0
		while data:
			nnoutput = self.net.forward(data[0])
			loss = self.loss_function.forward(nnoutput, data[1])
			sum_loss += loss
			gradInput = self.loss_function.backward(data[1])
			
			self.net.backward(gradInput)
			data = self.dataloader.feed() 

		print "avg loss: %f"%(sum_loss/(float)(self.dataloader.traindatasize))
		self.dataloader.reset()

	def get_train_acc(self):
		correct = 0.0
		data = self.dataloader.feed()
		while data:
			nnoutput = self.net.forward(data[0])
			
			if np.argmax(nnoutput) == np.argmax(data[1]):
				correct += 1.0
			data = self.dataloader.feed()

		self.dataloader.reset()
		return (correct/(float)(self.dataloader.traindatasize))
		
	def get_val_acc(self):
		correct = 0.0
		data_list = self.dataloader.get_val_datapair()
		for data in data_list:
			nnoutput = self.net.forward(data[0])
			if np.argmax(nnoutput) == np.argmax(data[1]):
				correct += 1.0
		return (correct/(float)(len(data_list)))
		
		
