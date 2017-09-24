import numpy as np
from dataloader import dataloader
from loss import Softmax_Cross_entropy
from layer import Linear
from nn import Net
from trainer import Trainer
import pdb

#pdb.set_trace()

epoch = 1000

train_data_path = '../data/nolabel_train.txt'
train_label_path = '../data/trainlabel'
val_data_path = '../data/nolabel_val.txt'
val_label_path = '../data/validlabel'
test_data_path = '../data/nolabel_test.txt'
test_label_path = '../data/testlabel'

def main():
	train_loader = dataloader(train_data_path, train_label_path, val_data_path, val_label_path)
	net = Net()

	nn_input_size = train_loader.traindata.shape[1]
	num_of_class = 10

	l1 = Linear(nn_input_size, 100)
	l2 = Linear(100, num_of_class)

	net.add(l1)
	net.add(l2)


	loss = Softmax_Cross_entropy()

	trainer = Trainer(0, train_loader, None, loss, net)

	for i in range(epoch):
		print "epoch %d"%i 
		trainer.train()
		print "train acc: %f "%trainer.get_train_acc()
		print "val acc: %f"%trainer.get_val_acc()


if __name__ == '__main__':
	main()