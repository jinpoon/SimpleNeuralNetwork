import numpy as np
from abc import ABCMeta, abstractmethod
from enum import Enum
import math

eps = 1e-34

class Layer(object): #abstract class of neural layers
	def __init__(self):
		self._is_training = True

	@abstractmethod
	def forward(self, input):
		return

	@abstractmethod
	def backward(self, gradInput):
		return 
		
class OutputLayer(Layer):
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
	def __init__(self, numInput, numOutput, lr=0.1, momentum=0.0):
		super(Linear, self).__init__()
		self.numInput = numInput
		self.numOutput = numOutput

		rg = math.sqrt(6) / (numOutput + numInput);
		self.weight = np.random.rand(numOutput, numInput) * 2 * rg - rg;
		self.bias = np.zeros(shape=(numOutput, 1));
		self.lastoutput = []
		self.lastinput = []
		self.lr = lr

		self.wv = np.zeros((numOutput, numInput))
		self.bv = np.zeros((numOutput, 1))
		self.momentum = momentum

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

		gradient_b = np.sum(gradInput, 1).reshape((self.numOutput, 1))

		gradient_w = np.dot(gradInput, np.transpose(self.lastinput))
		gradOutput = np.dot(np.transpose(self.weight), gradInput)

		self.update(gradient_w, gradient_b)

		return gradOutput

	def update(self, gradient_w, gradient_b):
		self.wv = -self.lr * gradient_w + self.momentum * self.wv
		self.weight = self.weight + self.wv
		self.bv = -self.lr * gradient_b + self.momentum * self.bv
		self.bias = self.bias + self.bv
	
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
		elif self.activation == 'relu':
			self.lastoutput = input
			self.lastoutput[input < 0] = 0
		else:
			self.lastoutput = []
		return self.lastoutput

	def backward(self, gradInput):
		if self.activation == 'sigmoid':
			return self.lastoutput * (1 - self.lastoutput) * gradInput
		elif self.activation == 'relu':
			gradInput[self.lastoutput == 0] = 0
			return gradInput
		else:
			return []
		


class BatchNorm(Layer):
	"""docstring for BatchNorm"""
	def __init__(self, numInput):
		super(BatchNorm, self).__init__()
		self.numInput = numInput
		self.numOutput = numInput

		self.gamma = np.ones((numInput, 1))
		self.beta = np.zeros((numInput, 1))
		self.lastinput = []
		self.lastxhat = []
		self.lr = 0.1 #learning rate
		#self.momentum = 0.1

	def forward(self, input):
		if self._is_training == False:
			u, b, var, xh = self.cache
			xh = (input-u) / np.sqrt(var + eps)
			return self.gamma * xh + self.beta


		self.lastinput = input
		D, bz = input.shape

		u = (1. / bz) * (np.sum(input, 1)).reshape(D, 1)
		b = input - u
		var = (1. / bz) * (np.sum(b*b, 1)).reshape(D, 1)

		xh = b / np.sqrt(var + eps)
		'''print b
		print np.sqrt(var + eps)
		print xh
		exit(1)'''

		self.cache = (u, b, var, xh)

		return self.gamma * xh + self.beta

	def backward(self, gradInput):
		D, bz = gradInput.shape
		u, b, var, xh = self.cache 

		den = var + eps

		grad_beta = (np.sum(gradInput, 1)).reshape(D, 1)
		grad_gamma = np.sum(xh * gradInput, 1).reshape(D, 1)

		grad_h = gradInput * self.gamma
		#print grad_h
		
		gradOutput =(1./bz) * (1./np.sqrt(den)) * (bz*grad_h - np.sum(grad_h, 1).reshape(D, 1) - 
			xh * (np.sum(grad_h * xh, 1).reshape(D, 1))) 
		
		self.update(grad_gamma, grad_beta)
		return gradOutput


	def update(self, grad_g, grad_b):
		#print grad_g.shape
		self.gamma = self.gamma - self.lr * grad_g
		#print self.gamma.shape
		self.beta = self.beta - self.lr * grad_b




		
