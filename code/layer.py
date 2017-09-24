import numpy as np
from abc import ABCMeta, abstractmethod
from enum import Enum
import math

#class Activation(Enum):
#	noActivation = 1
#	Sigmoid = 2
#	ReLU = 3
#	Tanh = 4

class Layer(object): #abstract class of neural layers
	def __init__(self):
		pass

	@abstractmethod
	def forward(self, input):
		return

	@abstractmethod
	def backward(self, gradInput):
		return 
		
class OutputLayer(Layer):
	"""abstract class for output layer"""
	def __init__(self):
		super(OutputLayer, self).__init__()
		self.input = []
		self.lastoutput = []
	
	@abstractmethod
	def form_output(self, loss):
		return

	@abstractmethod
	def form_grads(self, gradInput):
		return
		
	def forward(self, input):
		return self.form_output(input)

	def backward(self, gradInput):
		return self.form_grads(gradInput);

class Linear(Layer): #linear layer, contain

	def __init__(self, numInput, numOutput):
		super(Linear, self).__init__()
		self.numInput = numInput
		self.numOutput = numOutput

		rg = math.sqrt(6) / (numOutput + numInput);
		self.weight = np.random.rand(numOutput, numInput) * 2 * rg - rg;
		self.bias = np.zeros(shape=(numOutput, 1));
		self.lastoutput = []
		self.lastinput = []
		self.lr = 0.1

	def set_learning_rate(self, lr):
		self.lr = lr

	def activate(self, x):
		return 1/(1 + np.exp(-x))

	def forward(self, input):
		assert input.shape[0] == self.numInput
		x = np.dot(self.weight, input) + self.bias
		self.lastoutput = x
		self.lastinput = input
		return self.lastoutput
		
	def backward(self, gradInput):
		assert gradInput.shape[0] == self.numOutput

		gradient_b = np.mean(gradInput, 1).reshape((self.numOutput, 1))

		gradient_w = np.dot(gradInput, np.transpose(self.lastinput))  #might cause problem with batch
		gradOutput = np.dot(np.transpose(self.weight), gradInput)
		#print gradOutput

		self.update(gradient_w, gradient_b)

		return gradOutput

	def update(self, gradient_w, gradient_b):
		self.weight = self.weight - self.lr * gradient_w
		#print self.bias
		self.bias = self.bias - self.lr * gradient_b
		#print self.bias
		#exit(1)
	
class Activation(Layer):
	"""layer of activation"""
	def __init__(self, activation="sigmoid", size=-1):
		super(Activation, self).__init__()
		self.activation = activation
		self.lastoutput = []

		self.numInput = self.numOutput = size #to meet the assertion in nn.add()

	def forward(self, input):
		if self.activation == 'sigmoid':
			self.lastoutput = 1/(1 + np.exp(-input))
		else:
			self.lastoutput = []
		return self.lastoutput

	def backward(self, gradInput):
		if self.activation == 'sigmoid':
			return self.lastoutput * (1 - self.lastoutput) * gradInput
		else:
			return []
		


class BatchNorm(Layer):
	"""docstring for BatchNorm"""
	def __init__(self):
		super(BatchNorm, self).__init__()

	def forward(self, input):
		pass

	def backward(self, input):
		pass
		

'''
class Softmax_CrossLayer(OutputLayer):
	"""docstring for SoftmaxLayer"""
	def __init__(self, inputsize):
		super(SoftmaxLayer, self).__init__()
		self.numInput = inputsize
		self.inputsize = inputsize
		self.lastoutput = []

	def form_output(self, intput):
		self.lastoutput =  np.exp(intput) /np.sum(np.exp(intput))
		return self.lastoutput

	def form_grads(self, gradInput):
		gradOutput = np.zeros(shape=(self.inputsize,1))
		for i in range(0, self.inputsize):
			for j in range(0, self.inputsize):
				if i==j:
					gradOutput[i, 0] += self.lastoutput[i, 0] * (1 - self.lastoutput[i, 0])
				else:
					gradOutput[i, 0] += -self.lastoutput[i, 0] * self.lastoutput[j, 0]

		gradOutput = gradOutput*gradInput
		return gradOutput
'''



		
