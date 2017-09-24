import numpy as np
from abc import ABCMeta, abstractmethod
from layer import OutputLayer

class Optimizer(object):

	def __init__(self, batchSize, learning_rate, loss_function):
		pass
		self.lr = learning_rate
		self.batchSize = batchSize
		self.loss_function = loss_function

class Net(object):
	def __init__(self):
		#super(Net, self).__init__()
		self.layers = []
		self.last_output = []

	def add(self, layer):
		if len(self.layers) and layer.numInput != self.layers[-1].numOutput:
			print 'dimension of layer not match!'
			exit(1)

		self.layers.append(layer)

	def forward(self, input):
		output = input
		for i in range(0,len(self.layers)):
			output = self.layers[i].forward(output)

		self.last_output = output

		return self.last_output


	def backward(self, gradInput):
		gradOutput = gradInput
		for layer in self.layers[::-1]:
			gradOutput = layer.backward(gradOutput)
			#print gradOutput
		#print (self.layers[0].weight)[0,0]
		return gradOutput
