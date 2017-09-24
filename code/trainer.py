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
		batch = self.dataloader.feed()
		sum_loss = 0.0
		nb_of_class = 10
		while batch:
			#print "batch training"
			batchx = batch[0]
			batchy= batch[1]

			nnoutput = self.net.forward(batchx)
			loss = self.loss_function.forward(nnoutput, batchy)
			sum_loss += np.sum(loss)
			grads = self.loss_function.backward(batchy)

			self.net.backward(grads)
			batch = self.dataloader.feed() 

		print "avg loss: %f"%(sum_loss/(float)(self.dataloader.traindatasize))
		self.dataloader.reset()

	def get_train_acc(self):
		correct = 0.0
		batch = self.dataloader.feed()
		while batch:
			batchx = batch[0]
			batchy= batch[1]
			
			nnoutput = self.net.forward(batchx)
			correct += np.sum(np.argmax(nnoutput, 0) == np.argmax(batchy, 0));
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
		
		
