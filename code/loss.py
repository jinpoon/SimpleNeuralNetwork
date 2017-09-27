import numpy as np
from abc import ABCMeta, abstractmethod

class loss_function(object):
	"""docstring for loss_function"""
	def __init__(self):
		pass

	@abstractmethod
	def forward(self, nnoutput, target):
		return

	@abstractmethod
	def backward(self, target):
		return

class Softmax_Cross_entropy(loss_function):
	"""docstring for cross_entropy"""
	def __init__(self):
		super(Softmax_Cross_entropy, self).__init__()
		self.last_nnoutput = []
		self.last_loss = 0
		
	def forward(self, nnoutput, target):
		loss = -np.sum(target * np.log(np.exp(nnoutput) /np.sum(np.exp(nnoutput), 0)), 0)
		self.last_nnoutput = nnoutput
		return loss
		
	def backward(self, target):
		return -(target - self.last_nnoutput)
