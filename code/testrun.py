import numpy as np
from dataloader import dataloader
from loss import Softmax_Cross_entropy
from layer import Linear, Activation, BatchNorm
from nn import Net
from trainer import Trainer
import pdb
import matplotlib.pyplot as plt

#pdb.set_trace()
colorstr = ['red', 'blue', 'green', 'black']

epoch = 60

train_data_path = '../data/nolabel_train.txt'
train_label_path = '../data/trainlabel'
val_data_path = '../data/nolabel_val.txt'
val_label_path = '../data/validlabel'
test_data_path = '../data/nolabel_test.txt'
test_label_path = '../data/testlabel'

def main():

	numn= [20, 100, 200, 500]
	plt.figure(num = 0)

	for j in range(len(numn)):
		train_loader = dataloader(train_data_path, train_label_path, val_data_path, val_label_path, batchsize=1)
		net = Net()

		nn_input_size = train_loader.traindata.shape[1]
		num_of_class = 10

		l1 = Linear(nn_input_size, numn[j], lr=0.01, momentum=0.5)
		a1 = Activation('sigmoid', numn[j])
		l2 = Linear(numn[j], num_of_class, lr=0.01, momentum=0.5)
		a2 = Activation('sigmoid', num_of_class)

		net.add(l1)
		#net.add(b1) #add batch normalization
		net.add(a1)
		net.add(l2)
		#net.add(b2) #add batch normalization
		net.add(a2)

		loss = Softmax_Cross_entropy()

		trainer = Trainer(train_loader, None, loss, net)

		x = []
		vl = []
		ve = []

		for i in range(epoch):
			print "epoch %d"%i 
			training_loss = trainer.train()
			x.append(i+1)
			ve.append(1 - trainer.get_val_acc())
		plt.plot(x, ve, color=colorstr[j], linewidth=1.0, linestyle='--')

	plt.show()

	"""
	plt.figure(num = 0)
	for j in range(len(momentum)):
		train_loader = dataloader(train_data_path, train_label_path, val_data_path, val_label_path, batchsize=1)
		net = Net()

		nn_input_size = train_loader.traindata.shape[1]
		num_of_class = 10

		l1 = Linear(nn_input_size, 100, momentum=momentum[j])
		b1 = BatchNorm(100)
		a1 = Activation('sigmoid', 100)
		l2 = Linear(100, num_of_class, momentum=momentum[j])
		b2 = BatchNorm(num_of_class)
		a2 = Activation('sigmoid', num_of_class)

		net.add(l1)
		#net.add(b1) #add batch normalization
		net.add(a1)
		net.add(l2)
		#net.add(b2) #add batch normalization
		net.add(a2)

		loss = Softmax_Cross_entropy()

		trainer = Trainer(train_loader, None, loss, net)

		x = []
		ve = []

		for i in range(epoch):
			print "epoch %d"%i 
			training_loss = trainer.train()
			x.append(i+1)
			ve.append(1 - trainer.get_val_acc())
		plt.plot(x, ve, color=colorstr[j], linewidth=1.0, linestyle='--')

	plt.show()
	"""


if __name__ == '__main__':
	main()