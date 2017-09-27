# SimpleNeuralNetwork

This repository contains implementation of a simple feed-forward NN(code/) and a tiny dataset(data/).

Layer is a class that contains forward and backward function. forward(input) takes input from the last
layer and returns output of this layer. backward(gradInput) takes backward result from the last layer
and returns gradOutput of this layer.

Net is a class of the overall model. It maintains a list of layers controls how they forward and backword.
Trainer manages the net object and the loss function.

The data is the tiny subset of MNIST, where the data is a 784-dim vector (represents a 28*28 matrix) and
the target is the number(0-9)

See code/run.py to run the network. 
