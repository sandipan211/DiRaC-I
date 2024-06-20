import numpy as np
import argparse
from scipy import io, spatial
import time
from random import shuffle
import random
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import os
import sys
#new change
import pickle
parser = argparse.ArgumentParser(description="DeViSE")

parser.add_argument('-data', '--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='AWA2', type=str)
#new change - added next two params
parser.add_argument('-sn', '--sn', default=1, type=int, help='split number')
parser.add_argument('-al_lr', '--al_lr', default=0.01, type=float, help='learning rate used during active learning')
parser.add_argument('-e', '--epochs', default=100, type=int)
parser.add_argument('-es', '--early_stop', default=10, type=int)
parser.add_argument('-norm', '--norm_type', help='std(standard), L2, None', default='std', type=str)
parser.add_argument('-lr', '--lr', default=0.01, type=float)
parser.add_argument('-mr', '--margin', default=1, type=float)
parser.add_argument('-seed', '--rand_seed', default=42, type=int)
parser.add_argument('-al_seed', '--al_seed', default = 'new_seed_final', type =str)

"""
Best Values of (norm, lr, mr) found by validation & corr. test accuracies:
SUN  -> (None, 0.01, 3.0)-> Test Acc : 0.5569
CUB  -> (L2, 1.0, 1.0)   -> Test Acc : 0.4407
APY  -> (L2, 1.0, 1.0)   -> Test Acc : 0.3333
AWA1 -> (std, 0.01, 200) -> Test Acc : 0.5525
AWA2 -> (std, 0.001, 150)-> Test Acc : 0.5768
"""

class DeViSE():
	
	def __init__(self, args):

		self.args = args

		random.seed(self.args.rand_seed)
		np.random.seed(self.args.rand_seed)

		#new change - get the split_info in final results
		splits_folder = '/home/gdata/sandipan/BTP2021/' + args.dataset + '/' + 'split_info_lr' + str(args.al_lr) + '/'
		pklfile = splits_folder + 'u_split' + str(args.sn) + '_' + 'split_info_' + args.dataset + '.pickle'
		res = open(pklfile, 'rb')
		final_results = pickle.load(res)
		print('Split info: \n')
		print(final_results)

		data_folder = '/home/gdata/sandipan/BTP2021/xlsa17_final/data/'+args.dataset+'/'
		res101 = io.loadmat(data_folder+'res101.mat')

		#new change - get the common unseen classes
		# self.common_unseen = ['sheep', 'dolphin', 'bat', 'seal', 'blue+whale']
		self.common_unseen = final_results['common_unseen']
		print('Common unseen test classes ({}): {}'.format(len(self.common_unseen), self.common_unseen))

		if args.al_seed == 'original':
			att_splits = io.loadmat(data_folder + 'att_splits.mat')
		else:
			#new change - file name change
			split_name = 's' + str(args.sn)
			att_splits = io.loadmat(data_folder+'att_splits_' + args.dataset + '_al_' + args.al_seed + '_' + split_name + '.mat')

			print(data_folder+'att_splits_' + args.dataset + '_al_' + args.al_seed + '_' + split_name + '.mat')

		train_loc = 'train_loc'
		val_loc = 'val_loc'
		test_loc = 'test_unseen_loc'

		feat = res101['features']
		# Shape -> (dxN)
		self.X_train = feat[:, np.squeeze(att_splits[train_loc]-1)]
		self.X_val = feat[:, np.squeeze(att_splits[val_loc]-1)]
		self.X_test = feat[:, np.squeeze(att_splits[test_loc]-1)]

		print('Tr:{}; Val:{}; Ts:{}\n'.format(self.X_train.shape[1], self.X_val.shape[1], self.X_test.shape[1]))

		labels = res101['labels']
		self.labels_train = np.squeeze(labels[np.squeeze(att_splits[train_loc]-1)])
		self.labels_val = np.squeeze(labels[np.squeeze(att_splits[val_loc]-1)])
		self.labels_test = np.squeeze(labels[np.squeeze(att_splits[test_loc]-1)])

		train_labels_seen = np.unique(self.labels_train)
		val_labels_unseen = np.unique(self.labels_val)
		test_labels_unseen = np.unique(self.labels_test)

		all_names = att_splits['allclasses_names']
		self.trainclass_names, self.valclass_names, self.testclass_names = [], [], []
		for i in train_labels_seen:
			self.trainclass_names.append(all_names[i - 1][0][0])
		for i in val_labels_unseen:
			self.valclass_names.append(all_names[i - 1][0][0])
		for i in test_labels_unseen:
			self.testclass_names.append(all_names[i - 1][0][0])
		# all the labels are in order as given in classes.txt or allclasses_names(starting with label 1)

		#new change
		self.test_res = {}
		self.test_res['zsl_train'] = self.trainclass_names
		self.test_res['zsl_val'] = self.valclass_names
		self.test_res['zsl_test'] = self.testclass_names
		self.test_res['zsl_common_unseen'] = self.common_unseen

		print('\n\nTrain classes ({}): {}'.format(len(self.trainclass_names), self.trainclass_names))
		print('\n\nVal classes ({}): {}'.format(len(self.valclass_names), self.valclass_names))
		print('\n\nTest classes ({}): {}'.format(len(self.testclass_names), self.testclass_names))

		i=0
		for labels in train_labels_seen:
			self.labels_train[self.labels_train == labels] = i    
			i+=1
		
		j=0
		for labels in val_labels_unseen:
			self.labels_val[self.labels_val == labels] = j
			j+=1
		
		k=0
		for labels in test_labels_unseen:
			self.labels_test[self.labels_test == labels] = k
			k+=1

		sig = att_splits['att']
		# Shape -> (Number of attributes, Number of Classes)
		self.train_sig = sig[:, train_labels_seen-1]
		self.val_sig = sig[:, val_labels_unseen-1]
		self.test_sig = sig[:, test_labels_unseen-1]

		if self.args.norm_type=='std':
			print('Standard Scaling\n')
			scaler = preprocessing.StandardScaler()
			scaler.fit(self.X_train.T)

			self.X_train = scaler.transform(self.X_train.T).T
			self.X_val = scaler.transform(self.X_val.T).T
			self.X_test = scaler.transform(self.X_test.T).T

		if self.args.norm_type=='L2':
			print('L2 norm Scaling\n')
			self.X_train = self.normalizeFeature(self.X_train.T).T
			# self.X_val = self.normalizeFeature(self.X_val.T).T
			# self.X_test = self.normalizeFeature(self.X_test.T).T

	def normalizeFeature(self, x):
	    # x = N x d (d:feature dimension, N:number of instances)
	    x = x + 1e-10
	    feature_norm = np.sum(x**2, axis=1)**0.5 # l2-norm
	    feat = x / feature_norm[:, np.newaxis]

	    return feat

	def update_W(self, W, idx, train_classes):
		
		for j in idx:
			
			X_n = self.X_train[:, j]
			y_n = self.labels_train[j]
			y_ = train_classes[train_classes!=y_n]
			XW = np.dot(X_n, W)
			gt_class_score = np.dot(XW, self.train_sig[:, y_n])

			for i, label in enumerate(y_):
				score = self.args.margin+np.dot(XW, self.train_sig[:, label])-gt_class_score
				if score>0:
					Y = np.expand_dims(self.train_sig[:, y_n]-self.train_sig[:, label], axis=0)
					W += self.args.lr*np.dot(np.expand_dims(X_n, axis=1), Y)
					break # acc. to the authors, it was practical to stop after first margin violating term was found
		return W

	def fit(self):

		print('Training...\n')

		best_val_acc = 0.0
		best_tr_acc = 0.0
		best_val_ep = -1
		best_tr_ep = -1
		
		rand_idx = np.arange(self.X_train.shape[1])

		W = np.random.rand(self.X_train.shape[0], self.train_sig.shape[0])
		W = self.normalizeFeature(W.T).T

		train_classes = np.unique(self.labels_train)

		for ep in range(self.args.epochs):

			start = time.time()

			shuffle(rand_idx)

			W = self.update_W(W, rand_idx, train_classes)

			tr_acc = self.zsl_acc(self.X_train, W, self.labels_train, self.train_sig)			
			val_acc = self.zsl_acc(self.X_val, W, self.labels_val, self.val_sig)

			end = time.time()
			
			elapsed = end-start
			
			print('Epoch:{}; Train Acc:{}; Val Acc:{}; Time taken:{:.0f}m {:.0f}s\n'.format(ep+1, tr_acc, val_acc, elapsed//60, elapsed%60))
			
			if val_acc>best_val_acc:
				best_val_acc = val_acc
				best_val_ep = ep+1
				best_W = np.copy(W)
			
			if tr_acc>best_tr_acc:
				best_tr_ep = ep+1
				best_tr_acc = tr_acc

			if ep+1-best_val_ep>self.args.early_stop:
				print('Early Stopping by {} epochs. Exiting...'.format(self.args.epochs-(ep+1)))
				break

		print('\nBest Val Acc:{} @ Epoch {}. Best Train Acc:{} @ Epoch {}\n'.format(best_val_acc, best_val_ep, best_tr_acc, best_tr_ep))
		
		return best_W

	def zsl_acc(self, X, W, y_true, sig, testclass_names = None, testing = False): # Class Averaged Top-1 Accuarcy

		XW = np.dot(X.T, W)# N x k
		dist = 1-spatial.distance.cdist(XW, sig.T, 'cosine')# N x C(no. of classes)
		predicted_classes = np.array([np.argmax(output) for output in dist])
		cm = confusion_matrix(y_true, predicted_classes)
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		acc = sum(cm.diagonal())/sig.shape[1]

		if testing == True:
			print(np.unique(y_true))
			print(testclass_names)
			classwise_accs = {testclass_names[i]:a for i, a in enumerate(cm.diagonal().tolist())}

			# new change
			classwise_accs_common_unseen = {k:v for k, v in classwise_accs.items() if k in self.common_unseen}
			print(len(classwise_accs_common_unseen))
			acc_common_unseen = sum(classwise_accs_common_unseen.values()) / len(classwise_accs_common_unseen)

			return acc, classwise_accs, acc_common_unseen, classwise_accs_common_unseen

		return acc

	def evaluate(self):

		best_W = self.fit()

		print('Testing...\n')

		test_acc, classwise_accs, acc_common_unseen, classwise_accs_common_unseen = self.zsl_acc(self.X_test, best_W, self.labels_test, self.test_sig, self.testclass_names, testing = True)

		print('Test Acc = ', test_acc)
		print('Class-wise accuracies: ', classwise_accs)
		#new change
		print('Common unseen Test Acc = {:.4f}'.format(acc_common_unseen))
		print('Common unseen Class-wise accuracies: ', classwise_accs_common_unseen)
		pklfile2 = report_folder + fname + '_' + args.dataset  + '_' + args.al_seed + '_results.pickle'
		pkl = open(pklfile2, 'wb')
		self.test_res['total_acc'] = test_acc
		self.test_res['total_classwise'] = classwise_accs
		self.test_res['common_unseen_acc'] = acc_common_unseen
		self.test_res['common_unseen_classwise'] = classwise_accs_common_unseen
		pickle.dump(self.test_res, pkl)
		pkl.close()


if __name__ == '__main__':
	
	args = parser.parse_args()

	#new change - made separate report folders
	czsl_folder = '/home/gdata/sandipan/BTP2021/new_zsl_models/DEVISE/CZSL_al_lr' + str(args.al_lr) + '/' 
	report_folder = czsl_folder +'u_split' + str(args.sn) + '/'
	if not os.path.exists(czsl_folder):
		os.mkdir(czsl_folder)
	if not os.path.exists(report_folder):
		os.mkdir(report_folder)		

	fname = os.path.splitext(os.path.basename(sys.argv[0]))[0] 
	#new change
	result_filename = report_folder + fname + '_' + args.dataset  + '_' + args.al_seed + '_reports.txt'
	sys.stdout = open(result_filename, 'w')
	print('Dataset : {}\n'.format(args.dataset))

	clf = DeViSE(args)	
	clf.evaluate()

	#new change
	print('############################# DONE #################################')
	print('\n\nTest results: ', clf.test_res)
	sys.stdout.close()