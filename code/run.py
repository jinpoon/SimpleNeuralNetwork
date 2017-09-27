import numpy as np
from dataloader import dataloader
from loss import Softmax_Cross_entropy
from layer import Linear, Activation, BatchNorm
from nn import Net
from trainer import Trainer
import pdb
import matplotlib.pyplot as plt

#pdb.set_trace()

epoch = 200

train_data_path = '../data/nolabel_train.txt'
train_label_path = '../data/trainlabel'
val_data_path = '../data/nolabel_val.txt'
val_label_path = '../data/validlabel'
test_data_path = '../data/nolabel_test.txt'
test_label_path = '../data/testlabel'

def main():
	train_loader = dataloader(train_data_path, train_label_path, val_data_path, val_label_path, test_data_path, test_label_path, batchsize=32)
	net = Net()

	nn_input_size = train_loader.traindata.shape[1]
	num_of_class = 10

	l1 = Linear(nn_input_size, 100)
	b1 = BatchNorm(100)
	a1 = Activation('sigmoid', 100)
	l2 = Linear(100, num_of_class)
	b2 = BatchNorm(num_of_class)
	a2 = Activation('sigmoid', num_of_class)

	lm = Linear(100, 100)
	bm = BatchNorm(100)
	am = Activation('sigmoid', 100)	

	net.add(l1)
	net.add(b1) #add batch normalization
	net.add(a1)

	net.add(lm)
	net.add(bm)
	net.add(am)
	
	net.add(l2)
	net.add(b2) #add batch normalization
	net.add(a2)

	loss = Softmax_Cross_entropy()

	trainer = Trainer(train_loader, None, loss, net)

	x = []
	tl = []
	vl = []
	te = []
	ve = []

	for i in range(epoch):
		print "epoch %d"%i 
		training_loss = trainer.train()
		x.append(i+1)
		
		tl.append(training_loss)
		vl.append(trainer.get_val_loss())
		te.append(1 - trainer.get_train_acc())
		ve.append(1 - trainer.get_val_acc())
		print "val loss: %f"%vl[i]
		print "train error: %f "%te[i]
		print "val error: %f"%ve[i]
		#print "test loss: %f"%trainer.get_test_loss()
		#print "test error: %f"%(1-trainer.get_test_acc())
		#trainer.visualize()

if __name__ == '__main__':
	main()