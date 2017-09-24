import numpy as np
from abc import ABCMeta, abstractmethod
from enum import Enum
import math

class Activation(Enum):
	noActivation = 1
	Sigmoid = 2
	ReLU = 3
	Tanh = 4

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

	def __init__(self, numInput, numOutput, activation=Activation.Sigmoid):
		super(Linear, self).__init__()
		self.numInput = numInput
		self.numOutput = numOutput
		self.activation = activation

		rg = math.sqrt(6) / (numOutput + numInput);
		self.weight = np.random.rand(numOutput, numInput) * 2 * rg - rg;
		print self.weight
		self.bias = np.zeros(shape=(numOutput, 1));
		self.lastoutput = []
		self.lastinput = []
		self.lr = 0.1

	def set_learning_rate(self, lr):
		self.lr = lr

	def activate(self, x):
		if self.activation == Activation.Sigmoid:
			return 1/(1 + np.exp(-x))
		else:
			print "Didn't implement this activation yet!"

	def forward(self, input):
		assert input.shape == (self.numInput, 1)
		x = np.dot(self.weight, input) + self.bias
		self.lastoutput = self.activate(x)
		self.lastinput = input
		return self.lastoutput
		
	def backward(self, gradInput):
		assert gradInput.shape == (self.numOutput, 1)
		if self.activation == Activation.Sigmoid :
			d1 = self.lastoutput * (1 - self.lastoutput)
			gradient_b = gradInput* d1
			gradient_w = np.dot((gradInput* d1), np.transpose(self.lastinput))
			gradOutput = np.dot(np.transpose(self.weight), gradInput* d1)
			self.update(gradient_w, gradient_b)
			return gradOutput
		else:
			print "Not implemented activation! output zero gradients"
			return np.zeros((self.numOutput, self.Input))

	def update(self, gradient_w, gradient_b):
		self.weight = self.weight - self.lr * gradient_w
		self.bias = self.bias - self.lr * gradient_b
		
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



		
