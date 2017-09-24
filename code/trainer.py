from nn import Net, Optimizer
import numpy as np

class Trainer(object):
	"""docstring for Trainer"""
	def __init__(self, dataloader, optimizer, loss_function, net):
		self.dataloader = dataloader
		self.optimizer = optimizer
		self.loss_function = loss_function
		self.net = net

	def train(self):
		bz = (float)(self.dataloader.batchsize)
		batch = self.dataloader.feed()
		sum_loss = 0.0
		nb_of_class = 10
		while batch:
			batch_grad = np.zeros(shape = (nb_of_class, 1))
			for data in batch:
				nnoutput = self.net.forward(data[0])
				loss = self.loss_function.forward(nnoutput, data[1])
				sum_loss += loss
				gradInput = self.loss_function.backward(data[1])
				batch_grad = batch_grad + gradInput
			batch_grad = batch_grad / bz
			self.net.backward(batch_grad)
			batch = self.dataloader.feed() 

		print "avg loss: %f"%(sum_loss/(float)(self.dataloader.traindatasize))
		self.dataloader.reset()

	def get_train_acc(self):
		correct = 0.0
		bz = (float)(self.dataloader.batchsize)
		batch = self.dataloader.feed()
		while batch:
			for data in batch:
				nnoutput = self.net.forward(data[0])

				if np.argmax(nnoutput) == np.argmax(data[1]):
					correct += 1.0				

			batch = self.dataloader.feed() 
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
		
		
