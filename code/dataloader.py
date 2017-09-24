import numpy as np

class dataloader(object):
	"""dataloader"""
	def __init__(self, traindatapath, traintargetpath, 
		valdatapath=None, valtargetpath=None, testdatapath=None,
		 testtargetpath=None,  batchsize = 1):
		self.batchsize = batchsize
		self.traindata = np.loadtxt(traindatapath, delimiter=',')
		self.traintarget = np.loadtxt(traintargetpath, dtype='int64')
		self.traindatasize = len(self.traintarget)
		self.idx = 0
		self.featuresize = 784


		self.valdatapath = valdatapath
		self.valtargetpath = valtargetpath
		#self.testdata = np.loadtxt(testdatapath, delimiter',')
		#self.testtarget = np.loadtxt(testtargetpath, dtype='int64')
		#self.testtarget =len(self.testtarget)

	def permute(self):
		perm = np.random.permutation(len(self.traintarget))
		self.traindata = self.traindata[perm]
		self.traintarget =self.traintarget[perm]

	def feed(self):
		if self.idx + self.batchsize >= self.traindatasize:
			return
		batchx = np.zeros(shape=(self.featuresize, self.batchsize))
		batchy = np.zeros(shape=(10, self.batchsize))

		for i in range(self.idx, self.idx + self.batchsize):
			etarget = np.zeros(shape = (10, 1))
			etarget[self.traintarget[self.idx]] = 1
			batchx[:, i-self.idx] = (self.traindata[self.idx]).reshape(self.featuresize)
			batchy[:, i-self.idx] =  etarget.reshape(10)
			
		self.idx += self.batchsize
		return (batchx, batchy)

	def reset(self):
		perm = np.random.permutation(len(self.traintarget))
		self.traindata = self.traindata[perm]
		self.traintarget = self.traintarget[perm]
		self.idx = 0

	def get_val_datapair(self):
		if not self.valdatapath or not self.valtargetpath:
			print "val data path not exits"
			return

		valdata = np.loadtxt(self.valdatapath, delimiter=',')
		valtarget = np.loadtxt(self.valtargetpath, dtype='int64')
		valdatasize = len(valtarget)

		pair_list = []

		for i in range(0, valdatasize):
			etarget = np.zeros(shape = (10, 1))
			etarget[valtarget[i]] = 1
			tdata = (valdata[i].reshape(self.featuresize, 1), etarget)
			pair_list.append(tdata)

		return pair_list		
