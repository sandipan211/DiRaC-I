import numpy as np
import argparse
from scipy import io
from sklearn.metrics import confusion_matrix
import os
import sys
#new change
import pickle

parser = argparse.ArgumentParser(description="GZSL with ESZSL")

parser.add_argument('-data', '--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='AWA2', type=str)
#new change - added next two params

parser.add_argument('-sn', '--sn', default=1, type=int, help='split number')
parser.add_argument('-al_lr', '--al_lr', default=0.01, type=float, help='learning rate used during active learning')

parser.add_argument('-mode', '--mode', help='train/test, if test, set alpha, gamma to best values as given below', default='train', type=str)
parser.add_argument('-alpha', '--alpha', default=0, type=int)
parser.add_argument('-gamma', '--gamma', default=0, type=int)
parser.add_argument('-al_seed', '--al_seed', default = 'new_seed_final', type =str)

"""
Alpha --> Regularizer for Kernel/Feature Space
Gamma --> Regularizer for Attribute Space
Best Values of (Alpha, Gamma) found by validation & corr. test accuracies:
AWA1 -> (3, 0) -> Seen : 0.8684 Unseen : 0.0529 HM : 0.0998
AWA2 -> (3, 0) -> Seen : 0.8884 Unseen : 0.0404 HM : 0.0772
CUB  -> (3, 0) -> Seen : 0.5653 Unseen : 0.1470 HM : 0.2334
SUN  -> (3, 2) -> Seen : 0.2841 Unseen : 0.1375 HM : 0.1853
APY  -> (2, 0) -> Seen : 0.8107 Unseen : 0.0225 HM : 0.0439
"""

class ESZSL():
	
	def __init__(self):

		#new change - get the split_info in final results
		splits_folder = '/home/gdata/sandipan/BTP2021/' + args.dataset + '/' + 'split_info_lr' + str(args.al_lr) + '/'
		pklfile = splits_folder + 'u_split' + str(args.sn) + '_' + 'split_info_' + args.dataset + '.pickle'
		res = open(pklfile, 'rb')
		final_results = pickle.load(res)
		print('Split info: \n')
		print(final_results)

		data_folder = '/home/gdata/sandipan/BTP2021/xlsa17_final/data/'+args.dataset+'/'
		res101 = io.loadmat(data_folder+'res101.mat')
		if args.al_seed == 'original':
			att_splits = io.loadmat(data_folder + 'att_splits.mat')
		else:
			#new change - file name change
			split_name = 's' + str(args.sn)
			att_splits = io.loadmat(data_folder+'att_splits_' + args.dataset + '_al_' + args.al_seed + '_' + split_name + '.mat')

			print(data_folder+'att_splits_' + args.dataset + '_al_' + args.al_seed + '_' + split_name + '.mat')
		
		#new change - get the common unseen classes
		# self.common_unseen = ['sheep', 'dolphin', 'bat', 'seal', 'blue+whale']
		self.common_unseen = final_results['common_unseen']
		print('Common unseen test classes ({}): {}'.format(len(self.common_unseen), self.common_unseen))

		train_loc = 'train_loc'
		val_loc = 'val_loc'
		trainval_loc = 'trainval_loc'
		test_seen_loc = 'test_seen_loc'
		test_unseen_loc = 'test_unseen_loc'

		feat = res101['features']
		# Shape -> (dxN)
		self.X_trainval_gzsl = feat[:, np.squeeze(att_splits[trainval_loc]-1)]
		self.X_test_seen = feat[:, np.squeeze(att_splits[test_seen_loc]-1)]
		self.X_test_unseen = feat[:, np.squeeze(att_splits[test_unseen_loc]-1)]

		labels = res101['labels']
		self.labels_trainval_gzsl = np.squeeze(labels[np.squeeze(att_splits[trainval_loc]-1)])
		self.labels_test_seen = np.squeeze(labels[np.squeeze(att_splits[test_seen_loc]-1)])
		self.labels_test_unseen = np.squeeze(labels[np.squeeze(att_splits[test_unseen_loc]-1)])
		self.labels_test = np.concatenate((self.labels_test_seen, self.labels_test_unseen), axis=0)

		train_classes = np.unique(np.squeeze(labels[np.squeeze(att_splits[train_loc]-1)]))
		val_classes = np.unique(np.squeeze(labels[np.squeeze(att_splits[val_loc]-1)]))
		trainval_classes_seen = np.unique(self.labels_trainval_gzsl)
		self.test_classes_seen = np.unique(self.labels_test_seen)
		self.test_classes_unseen = np.unique(self.labels_test_unseen)
		test_classes = np.unique(self.labels_test) # All Classes of the dataset

		#new change
		all_names = att_splits['allclasses_names']
		self.testnames_unseen, self.trainclass_names, self.valclass_names, self.testnames_seen = [], [], [], []
		for i in train_classes:
			self.trainclass_names.append(all_names[i-1][0][0])

		for i in val_classes:
			self.valclass_names.append(all_names[i-1][0][0])

		for i in self.test_classes_seen:
			self.testnames_seen.append(all_names[i-1][0][0])		

		for i in self.test_classes_unseen:
			self.testnames_unseen.append(all_names[i-1][0][0])
		# all the labels are in order as given in classes.txt or allclasses_names(starting with label 1)

		#new change
		self.test_res = {}
		self.test_res['gzsl_train'] = self.trainclass_names
		self.test_res['gzsl_val'] = self.valclass_names
		self.test_res['gzsl_test_seen'] = self.testnames_seen
		self.test_res['gzsl_test_unseen'] = self.testnames_unseen
		self.test_res['gzsl_common_unseen'] = self.common_unseen

		print('\n\nTrain classes ({}): {}'.format(len(self.trainclass_names), self.trainclass_names))
		print('\n\nVal classes ({}): {}'.format(len(self.valclass_names), self.valclass_names))
		print('\n\nTest seen classes ({}): {}'.format(len(self.testnames_seen), self.testnames_seen))
		print('\n\nTest unseen classes ({}): {}'.format(len(self.testnames_unseen), self.testnames_unseen))


		print('gzsl test classes = ', test_classes.shape)


		train_gzsl_indices=[]
		val_gzsl_indices=[]

		for cl in train_classes:
			train_gzsl_indices = train_gzsl_indices + np.squeeze(np.where(self.labels_trainval_gzsl==cl)).tolist()

		for cl in val_classes:
			val_gzsl_indices = val_gzsl_indices + np.squeeze(np.where(self.labels_trainval_gzsl==cl)).tolist()

		train_gzsl_indices = sorted(train_gzsl_indices)
		val_gzsl_indices = sorted(val_gzsl_indices)
		
		self.X_train_gzsl = self.X_trainval_gzsl[:, np.array(train_gzsl_indices)]
		self.labels_train_gzsl = self.labels_trainval_gzsl[np.array(train_gzsl_indices)]
		
		self.X_val_gzsl = self.X_trainval_gzsl[:, np.array(val_gzsl_indices)]
		self.labels_val_gzsl = self.labels_trainval_gzsl[np.array(val_gzsl_indices)]

		# Train and Val are first separated to find the best hyperparamters on val and then to finally use them to train on trainval set.

		print('Tr:{}; Val:{}; Tr+Val:{}; Test Seen:{}; Test Unseen:{}\n'.format(self.X_train_gzsl.shape[1], self.X_val_gzsl.shape[1], 
			                                                                    self.X_trainval_gzsl.shape[1], self.X_test_seen.shape[1], 
			                                                                    self.X_test_unseen.shape[1]))

		i=0
		for labels in trainval_classes_seen:
			self.labels_trainval_gzsl[self.labels_trainval_gzsl == labels] = i    
			i+=1

		j=0
		for labels in train_classes:
			self.labels_train_gzsl[self.labels_train_gzsl == labels] = j
			j+=1

		k=0
		for labels in val_classes:
			self.labels_val_gzsl[self.labels_val_gzsl == labels] = k
			k+=1

		self.gt_train_gzsl = np.zeros((self.labels_train_gzsl.shape[0], len(train_classes)))
		self.gt_train_gzsl[np.arange(self.labels_train_gzsl.shape[0]), self.labels_train_gzsl] = 1

		self.gt_trainval = np.zeros((self.labels_trainval_gzsl.shape[0], len(trainval_classes_seen)))
		self.gt_trainval[np.arange(self.labels_trainval_gzsl.shape[0]), self.labels_trainval_gzsl] = 1

		sig = att_splits['att']
		# Shape -> (Number of attributes, Number of Classes)
		self.trainval_sig = sig[:, trainval_classes_seen-1]
		self.train_sig = sig[:, train_classes-1]
		self.val_sig = sig[:, val_classes-1]
		self.test_sig = sig[:, test_classes-1] # Entire Signature Matrix

	def find_W(self, X, y, sig, alpha, gamma):

		part_0 = np.linalg.pinv(np.matmul(X, X.T) + (10**alpha)*np.eye(X.shape[0]))
		part_1 = np.matmul(np.matmul(X, y), sig.T)
		part_2 = np.linalg.pinv(np.matmul(sig, sig.T) + (10**gamma)*np.eye(sig.shape[0]))

		W = np.matmul(np.matmul(part_0, part_1), part_2) # Feature Dimension x Number of Attributes

		return W

	def fit(self):

		print('Training...\n')

		best_acc = 0.0

		for alph in range(-3, 4):
			for gamm in range(-3, 4):
				W = self.find_W(self.X_train_gzsl, self.gt_train_gzsl, self.train_sig, alph, gamm)
				acc = self.zsl_acc(self.X_val_gzsl, W, self.labels_val_gzsl, self.val_sig)
				print('Val Acc:{}; Alpha:{}; Gamma:{}\n'.format(acc, alph, gamm))
				if acc>best_acc:
					best_acc = acc
					alpha = alph
					gamma = gamm

		print('\nBest Val Acc:{} with Alpha:{} & Gamma:{}\n'.format(best_acc, alpha, gamma))
		
		return alpha, gamma

	def zsl_acc(self, X, W, y_true, sig): # Class Averaged Top-1 Accuarcy

		class_scores = np.matmul(np.matmul(X.T, W), sig) # N x Number of Classes
		predicted_classes = np.array([np.argmax(output) for output in class_scores])
		cm = confusion_matrix(y_true, predicted_classes)
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		acc = sum(cm.diagonal())/sig.shape[1]

		return acc


	#new change - added last parameter
	def zsl_acc_gzsl(self, X, W, y_true, classes, sig, testing = False, unseen = False): # Class Averaged Top-1 Accuarcy

		class_scores = np.matmul(np.matmul(X.T, W), sig) # N x Number of Classes
		y_pred = np.array([np.argmax(output)+1 for output in class_scores])

		per_class_acc = np.zeros(len(classes))

		for i in range(len(classes)):
			is_class = y_true==classes[i]
			per_class_acc[i] = ((y_pred[is_class]==y_true[is_class]).sum())/is_class.sum()

		#new change
		if testing == True:
			print("gzsl", per_class_acc.tolist())
			if unseen == True:
				classwise_accs = {self.testnames_unseen[i]:a for i, a in enumerate(per_class_acc.tolist())}
				classwise_accs_common_unseen = {k:v for k, v in classwise_accs.items() if k in self.common_unseen}
				print(len(classwise_accs_common_unseen))
				acc_common_unseen = sum(classwise_accs_common_unseen.values()) / len(classwise_accs_common_unseen)
				return per_class_acc.mean(), classwise_accs, acc_common_unseen, classwise_accs_common_unseen

		return per_class_acc.mean()

	def evaluate(self, alpha, gamma):

		print('Testing...\n')

		best_W = self.find_W(self.X_trainval_gzsl, self.gt_trainval, self.trainval_sig, alpha, gamma) # combine train and val

		print('Seen:')
		acc_seen_classes = self.zsl_acc_gzsl(self.X_test_seen, best_W, self.labels_test_seen, self.test_classes_seen, self.test_sig, testing = True)
		
		print('Unseen:')
		#new change
		acc_unseen_classes, classwise_accs, acc_common_unseen, classwise_accs_common_unseen  = self.zsl_acc_gzsl(self.X_test_unseen, best_W, self.labels_test_unseen, self.test_classes_unseen, self.test_sig, testing = True, unseen = True)
		HM = 2*acc_seen_classes*acc_unseen_classes/(acc_seen_classes+acc_unseen_classes)

		print('U:{}; S:{}; H:{}'.format(acc_unseen_classes, acc_seen_classes, HM))
		print('acc common unseen: ', acc_common_unseen)
		print('classwise_accs_common_unseen: ', classwise_accs_common_unseen)
		considered_HM = 2*acc_seen_classes*acc_common_unseen/(acc_seen_classes+acc_common_unseen) # considering only common unseen classes
		print('considered HM: ', considered_HM)
		pklfile2 = report_folder + fname + '_' + args.dataset  + '_' + args.al_seed + '_results.pickle'
		pkl = open(pklfile2, 'wb')
		self.test_res['acc_unseen_classes'] = acc_unseen_classes
		self.test_res['acc_seen_classes'] = acc_seen_classes
		self.test_res['total_HM'] = HM
		self.test_res['classwise_accs_unseen'] = classwise_accs
		self.test_res['acc_common_unseen'] = acc_common_unseen
		self.test_res['classwise_accs_common_unseen'] = classwise_accs_common_unseen
		self.test_res['considered_HM'] = considered_HM
		pickle.dump(self.test_res, pkl)
		pkl.close()


if __name__ == '__main__':
	
	args = parser.parse_args()
	#new change - made separate report folders
	gzsl_folder = '/home/gdata/sandipan/BTP2021/new_zsl_models/ESZSL/GZSL_al_lr' + str(args.al_lr) + '/' 
	report_folder = gzsl_folder +'u_split' + str(args.sn) + '/'
	if not os.path.exists(gzsl_folder):
		os.mkdir(gzsl_folder)
	if not os.path.exists(report_folder):
		os.mkdir(report_folder)	

	fname = os.path.splitext(os.path.basename(sys.argv[0]))[0] 
	#new change
	result_filename = report_folder + fname + '_' + args.dataset  + '_' + args.al_seed + '_reports.txt'	
	sys.stdout = open(result_filename, 'w')
	print('Dataset : {}\n'.format(args.dataset))

	clf = ESZSL()
	
	if args.mode=='train': 
		args.alpha, args.gamma = clf.fit()
	
	clf.evaluate(args.alpha, args.gamma)
	#new change
	print('############################# DONE #################################')
	print('\n\nTest results: ', clf.test_res)
	sys.stdout.close()