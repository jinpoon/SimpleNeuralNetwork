# SimpleNeuralNetwork

====Sep 23rd====


This repository contains implementation of a simple feed-forward NN(code/) and a mini dataset(data/).

Layer is a class that contains forward and backward function. forward(input) takes input from the last
layer and returns output of this layer. backward(gradInput) takes backward result from the last layer
and returns gradOutput of this layer.

Net is a class of the overall model. It contains layers and loss function, controls how they forward and backword.

See code/run.py to use them. 
