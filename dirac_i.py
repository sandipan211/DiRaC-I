
'''
Author: Sandipan Sarma
Project date: May 2021

Inputs:
	dataset with (image, label) pairs. The image folder be preprocessed by running train_test_split.py first to be in format <Dataset>/Data/<one folder corresponding to each class>
	class-attribute matrix

Outputs:
	Stage 1: seed classes capturing initial domain diversity and rarity
	Stage 2: final training (seen) classes for evaluating any ZSL image classification model

'''

import numpy as np
import pandas as pd
import re
from math import floor, log, ceil
import random
import os
import pickle
import pdb
import time
import sys

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms
from torchvision import utils, models


import math

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.spatial import distance_matrix


from random import shuffle, sample
import scipy.io as sio
import argparse
import shutil


import load_semantics as sm
from plots_with_av_heatmap import *
import compute_att_coverage as attcov
import seed_construction as sc
from utils.al_utils import *
from utils.openmax_funcs import *
from utils.make_folders import *
from utils.evaluation import Evaluation



# function to create .mat files according to the obtained seen-unseen split from Active Learning
def convert_to_mat(al_classes, image_folder, outfile, ratio = 0.7, dataroot = 'xlsa17_final/data', dataset = 'SUN', image_embedding = 'res101', class_embedding = 'att', train_num = 580):
	ratio = float(ratio)

	classes = os.listdir(image_folder)
	classes = sorted(classes) # the organization for each of SUN and CUB is such that the features and labels are provided for classes in the lexicographic order of their names

	# randomly choosing train and val classes amongst seen classes
	shuffle(al_classes)
	train_classes = al_classes[:train_num]
	val_classes = al_classes[train_num:]

	print('\n\n\nConverting to .mat file......')
	print("\n\nTrain_classes ({}): {}".format(len(train_classes), train_classes))
	print("\n\nVal_classes ({}): {}".format(len(val_classes), val_classes))
	test_classes = []
	for c in classes:
		if c not in train_classes and c not in val_classes:
			test_classes.append(c)

	print("\n\nTest_classes ({}): {}".format(len(test_classes), test_classes))
	count = 0
	temp1_loc = []
	temp2_loc = []
	test_unseen_loc = []

	# All these indices are 1-indexed as is the standard convention
	for c in classes:
		images1 = os.listdir(os.path.join(image_folder, c))

		if c in train_classes:
			for i in range(1, len(images1) + 1):
				temp1_loc.append(count+i)

		elif c in val_classes:
			for i in range(1, len(images1) + 1):
				temp2_loc.append(count+i)

		else:
			for i in range(1, len(images1) + 1):
				test_unseen_loc.append(count+i)

		count += len(images1)

	# randomly choosing test_seen examples from both train and val classes

	shuffle(temp1_loc)
	train_loc = temp1_loc[:int(ratio * len(temp1_loc))]
	test_seen_loc1 = temp1_loc[int(ratio * len(temp1_loc)):]

	shuffle(temp2_loc)
	val_loc = temp2_loc[:int(ratio * len(temp2_loc))]
	test_seen_loc2 = temp2_loc[int(ratio * len(temp2_loc)):]

	test_seen_loc = test_seen_loc1 + test_seen_loc2
	shuffle(test_seen_loc)

	trainval_loc = train_loc + val_loc
	shuffle(trainval_loc)

	trainval_loc = np.array(trainval_loc)
	train_loc = np.array(train_loc)
	val_loc = np.array(val_loc)
	test_seen_loc = np.array(test_seen_loc)
	test_unseen_loc = np.array(test_unseen_loc)

	print('\n\n')
	print("train_loc", len(train_loc))
	print("val_loc", len(val_loc))
	print("trainval_loc", len(trainval_loc))
	print("test_seen_loc", len(test_seen_loc))
	print("test_unseen_loc", len(test_unseen_loc))

	# using the existing .mat files, for the dictionary keys, so as to make a compatible .mat file which can be used as it is, independently
	matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
	att = matcontent['att']
	original_att = matcontent['original_att']
	allclasses_names = matcontent['allclasses_names']

	d = {'allclasses_names' : allclasses_names, 'att' : att, 'original_att' : original_att, 'test_seen_loc' : test_seen_loc, 'test_unseen_loc' : test_unseen_loc, 'train_loc' : train_loc, 'trainval_loc' : trainval_loc, 'val_loc' : val_loc}
	
	# finally saved .mat file
	split_name = 's' + str(args.sn) + '_cq' + str(args.query_classes)
	sio.savemat(dataroot + "/" + dataset + "/" + class_embedding + "_splits_" + dataset + "_al_" + outfile + "_" + split_name + ".mat", d)

# Runs active learning given the dataset and seed classes
# Assumes dataset images are in the format <Dataset>/Data/<one folder corresponding to each class>
def active_learning(num_epochs, lr, dataset, att_df, imagenet_overlapping_classes, seed_classes, class_clusters_by_name, weibull_threshold, outfile):
	lr = float(lr)
	weibull_threshold = float(weibull_threshold)
	
	class_labels = {}
	count=0
	for dirs in sorted(os.listdir(dataset + '/Data')):
		if dirs not in unknown_unknown_testclasses:
			# not considering unknown unknown classes in AL
			class_labels[dirs] = count
			# class_labels[dirs] = -1*count
			count+=1

	print('\n\nClass labels alphabetical: ', class_labels)
	label_to_class = {v:k for k,v in class_labels.items()}

	def is_image(filename):
		return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png"])

	class IndexedDataset(Dataset):

		def __init__(self, dir_path, transform=None, test=False):
			'''
			Args:
			- dir_path (string): path to the directory containing images
			- transform (torchvision.transforms.) (default=None)
			- test (boolean): True for labeled images, False otherwise (default=False)
			'''

			self.dir_path = dir_path
			self.transform = transform

			self.classes_start = {}
			self.classes_end = {}

			feats = sio.loadmat(dataset + "_features.mat")
			self.features = []

			self.class_to_images = {} # contains the indexes of the images for each class
			for c in class_labels.keys():
				self.class_to_images[c] = []

			self.image_to_class = [] # stores the class corresponding to each image
			self.image_by_name = [] # stores filename corresponding to each image
			image_filenames = []
			for f in os.listdir(dir_path):
				if os.path.isfile(os.path.join(dir_path,f)) and is_image(os.path.join(dir_path,f)):
					c = (f.split('/')[-1]).split('@')[0]
					if c not in unknown_unknown_testclasses:
						n = (f.split('/')[-1]).split('@')[1].split('.')[0]
						self.class_to_images[c].append(len(image_filenames)) # index where this image is appended in next to next line
						self.image_to_class.append(c)
						self.image_by_name.append(n)
						self.features.append(feats[(f.split('/')[-1]).split('@')[1]])
						image_filenames.append(os.path.join(dir_path,f))

			# next 5 lines (and printing class labels on line 158) only for debugging - can be removed after you see for once that data is being loaded correctly
			self.image_filenames = image_filenames
			print('\nTotal images = ', len(self.image_filenames))
			for c in self.class_to_images.keys():
				print('imgs in {} = {}'.format(c, len(self.class_to_images[c])))
			assert len(self.features) == len(self.image_filenames)


			# We assume that in the beginning, the entire dataset is unlabeled, both train and test:
			if test:
				# The image's class is given by the first element when splitting the image file name on '@'
				# if dataset == 'SUN' or dataset == 'CUB':
				#   self.labels = [class_labels[(f.split('/')[-1]).split('@')[0]] for f in self.image_filenames]
				# self.labels = [class_labels[(f.split('/')[-1]).split('@')[0]] for f in self.image_filenames]
				self.labels = [-1]*len(self.image_filenames)    # seed ordered labels
				self.alphabetical_labels = [0]*len(self.image_filenames)    # labels as per ordering in class_labels
				self.unlabeled_mask = np.ones(len(self.image_filenames))
			else:
				self.labels =[-1]*len(self.image_filenames)
				self.alphabetical_labels = [0]*len(self.image_filenames)    # labels as per ordering in class_labels
				self.unlabeled_mask = np.ones(len(self.image_filenames))

		def __len__(self):
			return len(self.image_filenames)

		def __getitem__(self, idx):

			img_name = self.image_filenames[idx]
			image = Image.open(img_name).convert('RGB')

			if self.transform:
				image = self.transform(image)

			# print(image, image.shape)
			return image,self.labels[idx], idx, self.features[idx], self.alphabetical_labels[idx]

		# Display the image [idx] and its filename
		def display(self, idx):
			img_name = self.image_filenames[idx]
			print(img_name)
			img=mpimg.imread(img_name)
			imgplot = plt.imshow(img)
			plt.show()
			return

		# Set the label of image [idx] to 'new_label'
		def update_label(self, idx, new_label):
			self.labels[idx] = new_label
			# what about alphabetical labels??
			self.unlabeled_mask[idx] = 0
			return

		# Set the label of image [idx] to that read from its filename
		def label_from_filename(self, idx):
			# if dataset == 'SUN' or dataset == 'CUB':
			#   self.labels[idx] = class_labels[(self.image_filenames[idx].split('/')[-1]).split('@')[0]]
			self.labels[idx] = class_labels[(self.image_filenames[idx].split('/')[-1]).split('@')[0]]
			# what about alphabetical labels??
			self.unlabeled_mask[idx] = 0
			return

		def label_class_examples(self, s_label, a_label):
			for i in self.class_to_images[label_to_class[a_label]]:
				self.labels[i] = s_label
				self.alphabetical_labels[i] = a_label
				self.unlabeled_mask[i] = 0
	
	'''
	Each query strategy below returns a list of len=query_size with indices of
	samples that are to be queried.

	For our experiments, we have used the margin query

	Arguments:
	- model (torch.nn.Module): not needed for `random_query`
	- device (torch.device): not needed for `random_query`
	- dataloader (torch.utils.data.DataLoader)
	- query_size (int): number of samples to be queried for class labels (default=2)

	'''
	def random_query(data_loader, all_CRs, imgs_per_seen_class, query_size=10):

		sample_idx = []

		# Because the data has already been shuffled inside the data loader,
		# we can simply return the `query_size` first samples from it
		for batch in data_loader:

			_, _, idx, _, _ = batch
			sample_idx.extend(idx.tolist())

			if len(sample_idx) >= query_size:
				break

		return sample_idx[0:query_size]

	def least_confidence_query(model, device, data_loader, all_CRs, imgs_per_seen_class, query_size=10):

		confidences = []
		indices = []

		model.eval()

		with torch.no_grad():
			for batch in data_loader:

				data, _, idx, _, _ = batch
				logits = model(data.to(device))
				probabilities = F.softmax(logits, dim=1)

				# Keep only the top class confidence for each sample
				most_probable = torch.max(probabilities, dim=1)[0]
				confidences.extend(most_probable.cpu().tolist())
				indices.extend(idx.tolist())

		conf = np.asarray(confidences)
		ind = np.asarray(indices)
		sorted_pool = np.argsort(conf)
		# Return the indices corresponding to the lowest `query_size` confidences
		return ind[sorted_pool][0:query_size]

	def margin_query(model, device, data_loader, all_CRs, imgs_per_seen_class, query_size=2):

		margins = []
		indices = []

		model.eval()

		with torch.no_grad():
			for batch in data_loader:

				data, targets, idx, _, _ = batch
				logits = model(data.to(device))
				# print('\n\nLogits: {}'.format(logits))
				probabilities = F.softmax(logits, dim=1)
				# print('\n\nProbs: {}'.format(probabilities))

				# Select the top two class confidences for each sample
				toptwo, toptwoindices = torch.topk(probabilities, 2, dim=1)

				# print('\n\nGround truth: {}'.format(targets))

				# print('\n\nTop two indices: {}'.format(toptwoindices))

				# print('\n\nGround truth labels:')


				# print('\n\nTop two :'.format(logits))

				# Compute the margins = differences between the two top confidences
				differences = toptwo[:,0]-toptwo[:,1]
				margins.extend(torch.abs(differences).cpu().tolist())
				indices.extend(idx.tolist())

		margin = np.asarray(margins)
		index = np.asarray(indices)
		sorted_pool = np.argsort(margin)

		return index[sorted_pool][0:query_size]

	def most_ambiguous_query(dataset, model, device, data_loader, seed_classes, weibull_model, weibull_threshold, seedLabel_to_class, query_iter, dataset_name, imgs_per_seen_class, query_size=2):

		pred_vars = []
		indices = []
		k = 3
		min_candidates = max(5, ceil(3*np.log(avg_imgs_per_class)))
		print('min_candidates: ', min_candidates)
		batch_num = 1
		features = []
		im_names = []
		seed_ordered_classes = list(seedLabel_to_class.values()) + ['unseen']

		model.eval()

		ftr_extract_start = time.time()

		with torch.no_grad():
			for batch in data_loader:

				img_data, _, idx, _, _ = batch
				logits = model(img_data.to(device))
				features.append(logits)
				for i in idx.tolist():
					# print(i)
					indices.append(i)
					# indices contain the actual indices where these batch of image are located

		# Get the prediction features.
		indices = np.asarray(indices)
		# print('orig features len: ', len(features))
		
		# before this line, len(features) = num_unlabeled_imgs, each entry holds an output tensor of shape (1, num_seen_classes)
		features = torch.cat(features,dim=0).cpu().numpy()
		# after this line, features.shape = (num_unlabeled_imgs , num_seen_classes)

		# print('after torch cat features: ', features.shape)
		features = np.array(features)[:, np.newaxis, :]
		# after this line, features.shape = (num_unlabeled_imgs , 1, num_seen_classes)
		# print('after np.newaxis features: ', features.shape)
		print('unlabeled images: ', len(indices))

		print('\nTime taken to extract unlabeled features: ', time.time() - ftr_extract_start)

		pred_softmax, pred_openmax = [], []
		score_softmax, score_openmax = [], []
		ftr_mindists = []

		if len(seed_classes) < 10:
			weibull_alpha = len(seed_classes)
		else:
			weibull_alpha = 10
			#  sensitivity of OpenMax to total number of “top classes”. For more than 10 classes in training set, it is fixed to 10 (optimal value suggested by Openmax authors)
		categories = list(range(0, len(seed_classes)))   # list of current training labels
		openmax_timer = time.time()
		for num, ftr in enumerate(features):
			so, ss, all_dists = openmax(weibull_model, categories, ftr, weibull_alpha, eu_weight = 0.005, distance_type = "eucos")  # openmax_prob, softmax_prob
			pred_softmax.append(np.argmax(ss))

			pred_openmax.append(np.argmax(so) if np.max(so) >= weibull_threshold else len(seed_classes))
			score_softmax.append(ss)
			score_openmax.append(so)

			ftr_mindists.append(min(all_dists[0]))


		ftr_mindists = np.asarray(ftr_mindists) # dimension = (len(indices), ) = num of unlabeled img ftrs
		ftr_maxdists = np.argsort(-ftr_mindists) # gives the indices in max order
		candidate_idxs = indices[ftr_maxdists][0:min_candidates]

		print('Time taken for openmax predictions on unlabeled data: ', time.time() - openmax_timer)

		labels = [len(seed_classes)] * len(pred_openmax)
		eval_openmax = Evaluation(pred_openmax, labels, score_openmax)
		print('Openmax accuracy = {:.3f}'.format(eval_openmax.accuracy))
		print('Openmax F-measure = {:.3f}'.format(eval_openmax.f1_measure))

		candidate_labels = []
		candidate_names_per_iter = []
		for sample in candidate_idxs:
			label = class_labels[dataset.image_to_class[sample]]
			candidate_labels.append(label)

			# get candidate to plot
			img_name = dataset.image_filenames[sample]
			candidate_names_per_iter.append(img_name)

		plot_candidates(args, candidate_names_per_iter, query_iter, dataset.transform)

		# print('\ncandidate labels: ', candidate_labels)


		# visual plots of mavs till now and the maxmin candidates
		current_seed_names = list(seedLabel_to_class.values())
		current_candidate_names = [label_to_class[l] for l in candidate_labels]
		candidate_ftrs = features[ftr_maxdists][0:min_candidates] # shape = (min_candidates, 1, num_seen_classes)
		plot_visual_mavs(args, new_dirs, candidate_ftrs, weibull_model, categories, 'c', query_iter, current_seed_names, current_candidate_names)

		# after queried classes are found, visual plot is not done as final query is only on the basis of a class - not selecting any particular img feature in visual space - hence it is omitted for now



		unique_candidate_labels = list(set(candidate_labels))
		query_size = min(query_size, len(unique_candidate_labels))
		print('Query size = ', query_size)
		candidate_names = [label_to_class[l] for l in unique_candidate_labels]
		candidate_class_imps, att_ratios = get_importance(att_df, imgs_per_seen_class, candidate_names, label_to_class, class_labels)
		att_imp_all_iters[query_iter] = att_ratios


		candidate_scores = [candidate_class_imps[l] for l in candidate_labels]
		print('\ncandidate labels: ', candidate_labels)
		print('\ncandidate scores current: ', candidate_scores)

		print('\noverlapping: ', len(imagenet_overlapping_classes))
		added_classname = []


		if len(imagenet_overlapping_classes) > 0:
			overlapping_labels = [class_labels[c] for c in imagenet_overlapping_classes]
			priority_labels = [l for l in unique_candidate_labels if l in overlapping_labels]
			priority_allowed = min(query_size, len(priority_labels))
			priority_scores = [candidate_class_imps[l] for l in priority_labels]
			print('priority: ', priority_labels, end = '\t')
			print(priority_scores)

			for i in range(priority_allowed):
				maxscore = max(priority_scores)
				maxlabel = priority_labels[priority_scores.index(maxscore)]
				added_classname.append(label_to_class[maxlabel])
				print(maxlabel)
				priority_scores[:] = (value for value in priority_scores if value != maxscore)
				candidate_scores[:] = (value for value in candidate_scores if value != maxscore)
				priority_labels[:] = (value for value in priority_labels if value != maxlabel)
				candidate_labels[:] = (value for value in candidate_labels if value != maxlabel)

		while len(added_classname) < query_size:
			maxscore = max(candidate_scores)
			maxlabel = candidate_labels[candidate_scores.index(maxscore)]
			added_classname.append(label_to_class[maxlabel])
			print(maxlabel)
			candidate_scores[:] = (value for value in candidate_scores if value != maxscore)
			candidate_labels[:] = (value for value in candidate_labels if value != maxlabel)



		return added_classname, candidate_names_per_iter

	'''
	Queries the oracle (user, if `interactive` is set to True) for  labels for'query_size'
	samples using 'query_strategy'

	Arguments:
	- model (torch.nn.Module)
	- device: torch.device (CPU or GPU)
	- dataset (torch.utils.data.DataSet)
	- query_size (int): number of samples to be queried for class labels (default=2)
	- query_strategy (string): one of ['random', 'least_confidence', 'margin'],
							   otherwise defaults to 'random'
	- interactive (bool): if True, prompts the user to input the images' labels manually
						  if False, reads the labels from filenames (default=True)
	- pool_size (int): when > 0, the size of the randomly selected pool from the unlabeled_loader to consider
					   (otherwise defaults to considering all of the associated data)
	- batch_size (int): default=32
	- num_workers (int): default=1

	Modifies:
	- dataset: edits the labels of samples that have been queried; updates dataset.unlabeled_mask
	'''

	def query_the_oracle(model, device, dataset, test_set, seed_classes, weibull_model, weibull_threshold, seedLabel_to_class, query_iter, dataset_name, imgs_per_seen_class, query_size=2, query_strategy='margin', interactive=False, pool_size=0, batch_size=32, num_workers=1):

		query_start = time.time()
		unlabeled_idx = np.nonzero(dataset.unlabeled_mask)[0]

		# Select a pool of samples to query from
		if pool_size > 0:
			pool_idx = random.sample(range(1, len(unlabeled_idx)), pool_size)
			pool_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,sampler=SubsetRandomSampler(unlabeled_idx[pool_idx]))
		else:
			pool_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,sampler=SubsetRandomSampler(unlabeled_idx))

		# if query_strategy == 'margin':
		# 	sample_idx = margin_query(model, device, pool_loader, seed_classes, query_size)
		# elif query_strategy == 'least_confidence':
		# 	sample_idx = least_confidence_query(model, device, pool_loader, seed_classes, query_size)
		# elif query_strategy == 'most_ambiguous_query':
		added_classname, candidate_names_per_iter = most_ambiguous_query(dataset, model, device, pool_loader, seed_classes, weibull_model, weibull_threshold, seedLabel_to_class, query_iter, dataset_name, imgs_per_seen_class, query_size)

		# else:
		# 	sample_idx = random_query(pool_loader, seed_classes, query_size)
		print('added_classname: ', added_classname)
		unique_added_ordered = []
		for n in added_classname:
			if n not in unique_added_ordered:
				unique_added_ordered.append(n)

		# print('unique = ', unique_added_ordered)
		next_label = len(seed_classes)
		for l, n in enumerate(unique_added_ordered):
			# print('next label: ', next_label, end = '\t')
			dataset.label_class_examples(next_label + l, class_labels[n])
			test_set.label_class_examples(next_label + l, class_labels[n])

		print("\nnew: ", unique_added_ordered) # these labels correspond to classes which are to be included in the training classes in this iteration of active learning
		print('Time taken for query: ', time.time() - query_start)
		return unique_added_ordered, candidate_names_per_iter

	def train(model, device, train_loader, optimizer, criterion):

		model.train()
		bnum = 0
		epoch_loss = 0

		# print('seedLabel_to_class: ', seedLabel_to_class)
		# print('label_to_class: ', label_to_class)
		# print(len(train_loader))
		for batch in train_loader:

			data, target, _, _, a_target = batch
			# if bnum % 100 == 0:
			# 	uni, unicounts = torch.unique(target, return_counts = True)
			# 	a_uni, a_unicounts = torch.unique(a_target, return_counts = True)

			# 	print('seed_uni',uni)
			# 	print(unicounts)
			# 	print('al_uni', a_uni)
			# 	print(a_unicounts)

			data, target = data.to(device), target.to(device)
			optimizer.zero_grad()
			output = model(data)

			loss = criterion(output, target)
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()
			bnum += 1


		return epoch_loss

	def test(model, device, test_loader, criterion, total_test, display=False):
		model.eval()

		test_loss = 0
		n_correct = 0

		one = torch.ones(1, 1).to(device)
		zero = torch.zeros(1, 1).to(device)
		bnum = 0
		# print('seedLabel_to_class: ', seedLabel_to_class)
		# print('label_to_class: ', label_to_class)
		
		with torch.no_grad():
			for batch in test_loader:

				# print('test loader len: ',len(test_loader))

				data, target, _, _, a_target = batch
				# if bnum % 100 == 0:
				# 	uni, unicounts = torch.unique(target, return_counts = True)
				# 	a_uni, a_unicounts = torch.unique(a_target, return_counts = True)

				# 	print('seed_uni',uni)
				# 	print(unicounts)
				# 	print('al_uni', a_uni)
				# 	print(a_unicounts)

				data, target = data.to(device), target.to(device)

				output = model(data)
				test_loss += criterion(output, target).item()  # sum up batch loss
				prediction = output.argmax(dim=1, keepdim=True)
				torch.where(output.squeeze()<0.5, zero, one)  # get the index of the max log-probability
				n_correct += prediction.eq(target.view_as(prediction)).sum().item()
				bnum += 1

		test_loss /= total_test
		if display:
			print('Accuracy on the test set: ', (100. * n_correct / total_test))
		return test_loss, (100. * n_correct / total_test)



	np.random.seed(42)
	random.seed(10)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.manual_seed(999)

	batch_size = 32

	train_dir = dataset + '/train_dir'
	test_dir = dataset + '/test_dir'

	# device = torch.device("cuda")

	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda:0" if use_cuda else "cpu")
	print('Resource: Using ', device)

	print('Get train data')
	train_set = IndexedDataset(train_dir, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]))
	print('Get test data')
	test_set = IndexedDataset(test_dir, transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]), test=True)
	# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=1)

	print('Train set size: ', len(train_set))
	print('Test set size: ', len(test_set))

	# hyperparameter 'q' in paper
	query_size = cq
	# num_train_classes = all the classes being used for active learning
	num_train_classes = len(class_list)
	# parameter 'a' in paper
	avg_imgs_per_class = round((len(train_set) + len(test_set)) / num_train_classes)


	#########################      GET model       #########################
	# The classifier is a pre-trained ResNet101 with a random top layer dim = n_classes
	# classifier = models.resnet101(pretrained=True)
	classifier = models.resnet101(pretrained=False)
	classifier.load_state_dict(torch.load('./resnet101-5d3b4d8f.pth'))
	# freezing all layers except last one
	for param in classifier.parameters():
   		param.requires_grad = False

   	# Parameters of newly constructed modules have requires_grad=True by default
	num_ftrs = classifier.fc.in_features
	classifier.fc = nn.Linear(num_ftrs, len(seed_classes))
	# each time the head of CNN will have one node for every currently seen class
	classifier = classifier.to(device)

	criterion = nn.CrossEntropyLoss()

	# Observe that only parameters of final layer are being optimized.
	optimizer = optim.SGD(classifier.fc.parameters(), lr=lr, momentum=0.9, dampening=0, weight_decay=0.0001)
	# Decay LR by a factor of 0.1 every step_size epochs
	# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)	
	##########################################################################

	# get new labels as per the class ordering in seed_classes
	seedOrderedLabel = {cname:seed_idx for seed_idx, cname in enumerate(seed_classes)}
	seedLabel_to_class = {v:k for k,v in seedOrderedLabel.items()}
	# Label the initial subset (both train and test) corresponding to the seed classes provided
	for seed_label, alpha_label in [(seedOrderedLabel[cname], class_labels[cname]) for cname in seedOrderedLabel.keys()]:
		train_set.label_class_examples(seed_label, alpha_label)
		test_set.label_class_examples(seed_label, alpha_label)



	# Pre-train on the initial subset (train set)
	tr_labeled_idx = np.where(train_set.unlabeled_mask == 0)[0]
	print('train samples: ',len(tr_labeled_idx))
	tr_labeled_loader = DataLoader(train_set, batch_size=batch_size, num_workers=1, sampler=SubsetRandomSampler(tr_labeled_idx))
	# with corresponding test set
	te_labeled_idx = np.where(test_set.unlabeled_mask == 0)[0]
	print('test samples: ',len(te_labeled_idx))
	te_labeled_loader = DataLoader(test_set, batch_size=batch_size, num_workers=1, sampler=SubsetRandomSampler(te_labeled_idx))


	print('\n\n')
	epoch = 0
	delta = 1
	last_train_loss = 0
	train_loss = 1
	loss_drop_count = 0
	# stopping training just when test accuracy drops in succesive iterations might run into error for datasets which have very less number opf imgs per class. In such a case, like SUN which has only 20 imgs per class (avg.), it's test set will have only 2 images per class. So testing on them might result in disastrous test accuracies, which in turn will stop training very early. Since MAV computation depends on correctly classified training samples, a bad training might even result in some classes having no correct predictions, leading compute_MAVs into error. Hence, it's better to train for a fixed number of epochs

	train_start = time.time()

	while loss_drop_count < 3 :

		train_loss = train(classifier, device, tr_labeled_loader, optimizer, criterion)
		if epoch == 0:
			last_train_loss = train_loss
		# don't test the first epoch, cause some classes may have no predict samples, leading to error caused by compute_train_score_and_mavs_and_dists
		if epoch % 3 == 0 and epoch != 0:
			print("epoch {} train loss = {}".format(epoch, train_loss))
			_, current_test_acc = test(classifier, device, te_labeled_loader, criterion, len(te_labeled_idx), display=True)
			if last_train_loss - train_loss < delta:
				loss_drop_count += 1
				print('drop: ', loss_drop_count)

			last_train_loss = train_loss

		epoch += 1



	print('\nTime taken for training: ', time.time() - train_start)

	test(classifier, device, te_labeled_loader, criterion, len(te_labeled_idx), display=True)
	torch.save(classifier.state_dict(), dataset + '/' + outfile + '_u_split' + str(args.sn) + '_lr' + str(lr) + '_cq' + str(args.query_classes) + '.pth')


	# initializers

	query_iter = 0
	# all_CRs = {}
	prev_train_labels = []
	initial_seed_classes = seed_classes.copy()
	# loader_for_CRs = labeled_loader
	all_queried_classes = []
	seedOrderedLabel_iter = seedOrderedLabel
	seedLabel_to_class_iter = seedLabel_to_class
	tr_labeled_loader_iter = tr_labeled_loader
	te_labeled_loader_iter = te_labeled_loader

	candidates_all_iters = []
	att_imp_all_iters = {}

	while True:
		# Query the oracle for more labels
		print('\n\n\n################################################')
		query_iter += 1
		print('query iter {}'.format(query_iter))
		preprocessing_start = time.time()

		current_a_labels = [class_labels[cname] for cname in seedOrderedLabel_iter.keys()]
		# print('current_a_labels: ', current_a_labels)
		print('Computing MAVs for each seen class...\n')
		_, mavs, dists, imgs_per_seen_class = get_mavs_and_dists(classifier, device, tr_labeled_loader_iter, len(seed_classes), current_a_labels)
		seen_train_labels = [seed_idx for _, seed_idx in seedOrderedLabel_iter.items()]
		print('seen train labels: ', seen_train_labels)
		# EVT Meta-Recognition Calibration for Open Set Deep Networks, with per class Weibull fit to m largest distance to MAVs. 
		weibull_model = fit_weibull(mavs, dists, seen_train_labels)
		print('\nTime taken to compute MAVs and then weibull tailfitting: ', time.time() - preprocessing_start)

		new_classes, candidate_names_per_iter = query_the_oracle(classifier, device, train_set, test_set, seed_classes, weibull_model, weibull_threshold, seedLabel_to_class_iter, query_iter, dataset, imgs_per_seen_class, query_size=query_size, query_strategy='most_ambiguous_query', interactive=False, pool_size=0)
		new_unique = list(set(new_classes))
		imagenet_overlapping_classes = list(set(imagenet_overlapping_classes) - set(new_unique))
		# prev_train_labels = seen_train_labels.copy()

		candidates_all_iters.append(candidate_names_per_iter)
		print('new unique classes: ', new_unique)
		all_queried_classes.extend(new_unique)
		plot_tsne_clusters(dataset, class_list, att_df, class_clusters_by_name, initial_seed_classes, args, new_dirs, all_queried_classes, new_unique, query_iter)

		# pdb.set_trace()

		# extended_seed = seed_classes + new_unique

		# if class is not present in the current set of classes, append it
		for c in new_classes:
			if c not in seed_classes:
				seed_classes.append(c)

		print('Seed classes till now: ', seed_classes)

		# checking if seed class limit reached
		if (len(seed_classes) >= num_seen_classes): 
			# this is to get the whole ordering of all classes in the dataset, if number of seen classes is known apriori, then can stop at that number
			break

		################   Re-training a new model with increased classes #################
		del classifier

		# The classifier is a pre-trained ResNet101 with a random top layer dim = n_classes
		# classifier = models.resnet101(pretrained=True)
		# reload latest trained model

		classifier = models.resnet101(pretrained=False)
		classifier.load_state_dict(torch.load('./resnet101-5d3b4d8f.pth'))
		# freezing all layers except last one
		for param in classifier.parameters():
	   		param.requires_grad = False

	   	# Parameters of newly constructed modules have requires_grad=True by default
		num_ftrs = classifier.fc.in_features
		classifier.fc = nn.Linear(num_ftrs, len(seed_classes))
		# each time the head of CNN will have one node for every currently seen class
		classifier = classifier.to(device)

		criterion = nn.CrossEntropyLoss()
		# Observe that only parameters of final layer are being optimized.
		optimizer = optim.SGD(classifier.fc.parameters(), lr=lr, momentum=0.9, dampening=0, weight_decay=0.0001)
		# Decay LR by a factor of 0.1 every step_size epochs
		# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)	
		##########################################################################

		# get new labels as per the class ordering in seed_classes
		seedOrderedLabel_iter = {cname:seed_idx for seed_idx, cname in enumerate(seed_classes)}
		seedLabel_to_class_iter = {v:k for k,v in seedOrderedLabel_iter.items()}

		# Train the model on the data that has been labeled so far:
		tr_labeled_idx = np.where(train_set.unlabeled_mask == 0)[0]
		print('\n\ntrain samples: ',len(tr_labeled_idx))
		tr_labeled_loader = DataLoader(train_set, batch_size=batch_size, num_workers=1, sampler=SubsetRandomSampler(tr_labeled_idx))
		# with corresponding test set
		te_labeled_idx = np.where(test_set.unlabeled_mask == 0)[0]
		print('test samples: ',len(te_labeled_idx))
		te_labeled_loader = DataLoader(test_set, batch_size=batch_size, num_workers=1, sampler=SubsetRandomSampler(te_labeled_idx))


		epoch = 0
		delta = 1
		last_train_loss = 0
		train_loss = 1
		loss_drop_count = 0
		# stopping training just when test accuracy drops in succesive iterations might run into error for datasets which have very less number opf imgs per class. In such a case, like SUN which has only 20 imgs per class (avg.), it's test set will have only 2 images per class. So testing on them might result in disastrous test accuracies, which in turn will stop training very early. Since MAV computation depends on correctly classified training samples, a bad training might even result in some classes having no correct predictions, leading compute_MAVs into error. Hence, it's better to train for a fixed number of epochs

		train_start = time.time()

		while loss_drop_count < 3 :

			train_loss = train(classifier, device, tr_labeled_loader, optimizer, criterion)
			if epoch == 0:
				last_train_loss = train_loss
			# don't test the first epoch, cause some classes may have no predict samples, leading to error caused by compute_train_score_and_mavs_and_dists
			if epoch % 3 == 0 and epoch != 0:
				print("epoch {} train loss = {}".format(epoch, train_loss))
				_, current_test_acc = test(classifier, device, te_labeled_loader, criterion, len(te_labeled_idx), display=True)
				if last_train_loss - train_loss < delta:
					loss_drop_count += 1
					print('drop: ', loss_drop_count)

				last_train_loss = train_loss

			epoch += 1


		print('\nTime taken for training: ', time.time() - train_start)
		test(classifier, device, te_labeled_loader, criterion, len(te_labeled_idx), display=True)
		
		torch.save(classifier.state_dict(), dataset + '/' + outfile + '_u_split' + str(args.sn) + '_lr' + str(lr) + '_cq' + str(args.query_classes) + '.pth')
		# loader_for_CRs = labeled_loader
		tr_labeled_loader_iter = tr_labeled_loader
		te_labeled_loader_iter = te_labeled_loader



	return seed_classes, candidates_all_iters, att_imp_all_iters


def time_in_dhms(sectime):
	day = sectime // (24 * 3600)
	sectime = sectime % (24 * 3600)
	hour = sectime // 3600
	sectime %= 3600
	minutes = sectime // 60
	sectime %= 60
	seconds = sectime
	runtime = "Runtime: " + str(day) + "d: " + str(hour) + "h: " + str(minutes) + "m: " + str(seconds) +"s"
	return runtime


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Active Learning given seed classes and forming .mat files for Zero Shot Learning")
	parser.add_argument('-d','--dataset', default = 'SUN', help = 'AWA2, SUN, CUB')
	parser.add_argument('-prev_snum', '--prev_snum', type = int, default = -1, help='random unknown unknown split number to separate those classes and test hyperparameters')
	parser.add_argument('-snum', '--snum', type = int, default = -1, help='random unknown unknown split number for a new set of unknown unknown classes')
	parser.add_argument('-sn', '--sn', type = int, default = -1, help='split number will be decided based on other arguments. Should not set it explicitly')

	parser.add_argument('-cq', '--query_classes', type = int, default = 2, help='classes to add to seed set in each iteration of VSM')
		
	parser.add_argument('-es', '--num_epochs', type = int, default = 12, help='epochs to train')
	parser.add_argument('-c', '--re_clustering', action = 'store_true', help = 're-clustering and seed reconstruction')
	parser.add_argument('-l','--linkage', default = 'ward')  
	parser.add_argument('-lr', '--l_rate', default = 0.01, help = 'model learning rate')
	parser.add_argument('-m','--output_model', default = 'trained_model')
	parser.add_argument('-t', '--train_num', type = int, default = 580, help='number of train classes')
	parser.add_argument('-o', '--output_file', default = '_al', help = 'output mat file')
	parser.add_argument('-r', '--ratio_of_split', default = 0.8, help = 'ratio of training examples, remaining assigned as test_seen_classes')
	parser.add_argument('-n', '--num_seen_classes', type = int, default = 645, help='number of seen classes')
	parser.add_argument('-s', '--min_seeds', type = int, default = 5, help='minimum number of seed classes to pick')
	parser.add_argument('-w', '--weibull_threshold', default = 0.5, help='threshold for uncertainty-based rejection in Openmax')


	args = parser.parse_args()
	prog_start = time.time()

	dataset = args.dataset
	num_epochs = args.num_epochs
	cq = args.query_classes
	output_model = args.output_model
	train_num = args.train_num
	outfile = args.output_file
	ratio = args.ratio_of_split
	num_seen_classes = args.num_seen_classes
	re_clustering = args.re_clustering
	linkage = args.linkage
	lr = args.l_rate
	min_seeds = args.min_seeds
	weibull_threshold = args.weibull_threshold

	image_folder = dataset + '/Data'

	final_results = {}
	final_results['al_args'] = {
	'dataset':dataset,
	'prev_split_no_for_unk_unk':args.prev_snum,	
	'new_split_no_for_unk_unk':args.snum,
	'classes_per_AL_iter': args.query_classes,
	'num_epochs':num_epochs,
	'output_model':output_model,
	'train_num':train_num,
	'outfile':outfile,
	'ratio':ratio,
	'num_seen_classes':num_seen_classes,
	're_clustering':re_clustering,
	'linkage':linkage,
	'lr':lr,
	'min_seeds':min_seeds,
	'weibull_threshold':weibull_threshold,
	'image_folder':image_folder
	}


	# MODIFICATION: We can either work with a new randomly created set of common unseen classes (snum!= -1 and prev_snum=-1) or can test our hyperparameters on a previous set of common unseen classes, where we have to define which previous split we are using our common unseen classes from (snum=-1 and prev_snum!=-1) - define a common split num according to given arguments

	if (args.prev_snum != -1):
		print('Working with a previous set of common unseen classes taken from split number {}'.format(args.prev_snum))
		args.sn = args.prev_snum
	else:
		print('Working with a new set of common unseen classes, making it split number {}'.format(args.snum))
		args.sn = args.snum



	new_dirs = make_dirs(args)

	result_filename = new_dirs['reports_dir'] + 'u_split' + str(args.sn) + '_' + dataset + '_reports.txt'
	sys.stdout = open(result_filename, 'w')

	print('\n\nInitially created folders: \n')
	print(new_dirs)

	# get semantic matrix
	att_df, data_complete_info, imagenet_overlapping_classes, given_testclasses = sm.load_semantic_matrix(dataset = dataset)
	print('\nOriginally obtained overlapping classes: ', len(imagenet_overlapping_classes))
	final_results['universal_overlapping'] = imagenet_overlapping_classes

	if (args.snum != -1):
		# get new set of common unseen classes
		# sample half of the Xian test classes at random as unknown unknown and keep them separate from entire process- other test classes are used in seed construction and AL too
		while True:
			sample_num = int(len(given_testclasses)/2)
			unknown_unknown_testclasses = random.sample(given_testclasses, sample_num)
			# checking if common unseen classes are disjoint from our imagenet overlapping classes
			if len(set(imagenet_overlapping_classes).intersection(set(unknown_unknown_testclasses))) == 0:
				print('Common unseen randomly found are not among overlapping classes!')
				break

			print('Common unseen randomly found contain overlapping classes! Randomly choosing again...')

	elif args.prev_snum != -1:
		# update this if you add another hyperparam studies to folder names
		extra_foldername_for_hyperstudies = '_cq'+str(args.query_classes)+'/'
		# get unknown unknown classes from given split number
		pickle_file = new_dirs['split_info_dir'][:-len(extra_foldername_for_hyperstudies)] + '/u_split' + str(args.sn) + '_' + 'split_info_' + dataset + '.pickle'
		# We were doing hyperparam studies after we had already got results once for default cq values for CUB and SUN for all three splits. Since in new folder names there is '_cq'+str(args.query_classes)+'/' at the end which was not there in folder names of default run, we truncate as [:-len(extra_foldername_for_hyperstudies)] to obtain folder name of default run. However, if all the code is run afresh for all splits with default values of 2 and 4 for CUB and SUN again, then the new folder names would have that "extra_foldername_for_hyperstudies" at the end. Hence, after that if we want to use split info from that folder in the future, that time we dont need to truncate like [:-len(extra_foldername_for_hyperstudies)] - we can omit that much part.
		split_res = open(pickle_file, 'rb')
		given_split_results = pickle.load(split_res)

		# taking out the same unknown unknown classes of the given split number to perform hyperparameter sensitivity studies
		print('Loading common unseen classes from {}.........'.format(pickle_file))
		unknown_unknown_testclasses = given_split_results['common_unseen']
		final_results['common_unseen'] = unknown_unknown_testclasses
		print('\n\nCommon unseen classes ({}) taken from split: {}'.format(len(unknown_unknown_testclasses), unknown_unknown_testclasses))


	# remove unknown unknown from testclassees list as well as overlapping list
	usable_testclasses = list(set(given_testclasses) - set(unknown_unknown_testclasses))
	final_results['al_used_testclasses'] = usable_testclasses
	print('\n\nAL-usable unseen classes ({}) become: {}'.format(len(usable_testclasses), usable_testclasses))
	imagenet_overlapping_classes = list(set(imagenet_overlapping_classes) - set(unknown_unknown_testclasses))
	# remove unknown unknown class attributes from att_df
	att_df.drop(unknown_unknown_testclasses, axis = 0, inplace = True)


	all_overlappping = imagenet_overlapping_classes.copy()
	class_list = att_df.index.tolist()
	final_results['all_al_used_classes'] = class_list
	num_classes = att_df.shape[0]
	num_attributes = att_df.shape[1]  

	print(att_df)
	print('\nOverlapping after removing unknown unknown testclasses: {} \nOverlapping classes: {}\n\n'.format(len(imagenet_overlapping_classes), imagenet_overlapping_classes))
	print('\nClass list after removing unknown unknown testclasses: {} \nClass list: {}\n\n'.format(len(class_list), class_list))



	############################# seed construction phase #############################
	if re_clustering == True:

		# cluster all classes
		print('Re-clustering.....\n')

		clustered_classes, silhouette_avg_per_clustering = sc.cluster_classes(att_df, linkage, num_classes)
		plot_silhouette_score(silhouette_avg_per_clustering, dataset, args, new_dirs)
		class_clusters_by_name = sc.get_class_clusters(min_seeds, att_df, clustered_classes, silhouette_avg_per_clustering)
		print('Class clusters: {}\n'.format(class_clusters_by_name))

		seed_classes, cluster_att_info = sc.get_seeds(att_df, class_clusters_by_name)
		print('\n\nSeed classes: {}'.format(seed_classes))

		# obtain t-sne plot of classes color-coded as per their clusters
		plot_tsne_clusters(dataset, class_list, att_df, class_clusters_by_name, seed_classes, args, new_dirs)

		# store clustering results
		results = {
			'all_classes' : att_df.index,
			'all_atts' : att_df.columns,
			'cluster_combos' : clustered_classes,
			'sscore_avgs' : silhouette_avg_per_clustering,
			'clusters' : class_clusters_by_name,
			'seeds' : seed_classes,
			'cluster_att_info': cluster_att_info
		}

		final_results['clustering_results'] = results

		pickle_file = new_dirs['clusters_dir'] + 'u_split' + str(args.sn) + '_' + 'clustering_results_' + dataset + '.pickle'
		pkl = open(pickle_file, 'wb')
		pickle.dump(results, pkl)
		pkl.close()


	else:

		print('Using the latest clustered seeds.....')
		pickle_file = new_dirs['clusters_dir'] + 'u_split' + str(args.sn) + '_' + 'clustering_results_' + dataset + '.pickle'
		res = open(pickle_file, 'rb')
		clustering_results = pickle.load(res)

		clustered_classes, silhouette_avg_per_clustering = clustering_results['cluster_combos'], clustering_results['sscore_avgs']
		# plot_silhouette_score(silhouette_avg_per_clustering, dataset, args, new_dirs)
		class_clusters_by_name = clustering_results['clusters']
		print('\nClass clusters: {}'.format(class_clusters_by_name))

		seed_classes = clustering_results['seeds']
		print('\nSeed classes: {}'.format(seed_classes))
		final_results['clustering_results'] = clustering_results
		# obtain t-sne plot of classes color-coded as per their clusters
		# plot_tsne_clusters(dataset, class_list, att_df, class_clusters_by_name, seed_classes, args, new_dirs)

	print('\nFinished constructing seed set !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

	##################################################################################



	# remove already taken-in seed classes from overlapping classes
	imagenet_overlapping_classes = list(set(imagenet_overlapping_classes) - set(seed_classes))
	initial_seed_classes = seed_classes.copy()
	
	all_classes_after_AL, candidates_all_iters, att_imp_all_iters = active_learning(num_epochs, lr, dataset, att_df, imagenet_overlapping_classes, seed_classes, class_clusters_by_name, weibull_threshold, output_model)

	print('\n\n\nFinished Active Learning..............')
	print("\n\nClasses in active learning order", all_classes_after_AL)

	al_classes = all_classes_after_AL[:num_seen_classes]	
	test_classes = list(set(class_list) - set(al_classes))
	still_overlapping = list(set(all_overlappping).intersection(set(test_classes)))
	final_results['post_AL_trainval'] = al_classes
	final_results['post_AL_test'] = test_classes
	final_results['post_AL_overlapping'] = still_overlapping
	final_results['candidates_all_iters'] = candidates_all_iters
	final_results['att_imp_all_iters'] = att_imp_all_iters

	print('\n\n\nTrain classes: ', al_classes)
	print('\n\n\nTest classes: ', test_classes)
	print('\n\nStill overlapping: ', still_overlapping)



	if len(still_overlapping) > 0:

		nonoverlap = list(set(class_list) - set(all_overlappping))
		nonoverlap = list(set(nonoverlap) - set(initial_seed_classes))
		rep = list(set(al_classes).intersection(nonoverlap))

		rep_data = att_df.loc[rep]
		rem_overlap_data = att_df.loc[still_overlapping]
		dist_mat =  pd.DataFrame(distance_matrix(rem_overlap_data.values, rep_data.values), index=rem_overlap_data.index, columns=rep_data.index)
		swapped = {}
		for i in range(dist_mat.shape[0]):
			k = dist_mat.min(axis = 1).sort_values().idxmin()
			v = dist_mat.loc[k].idxmin()
			swapped[k] = v

			dist_mat.drop([v], axis = 1, inplace = True)
			dist_mat.drop([k], axis = 0, inplace = True)

		print('Swapped: ', swapped)
		final_results['swapped'] = swapped

		for k, v in swapped.items():
			alc_idx = al_classes.index(v)
			al_classes[alc_idx] = k
			te_idx = test_classes.index(k)
			test_classes[te_idx] = v

		print('\n\nFinal Training: ', al_classes)
		print('Final Testing: ', test_classes)
		print('Final Common Unseen Testing: ', unknown_unknown_testclasses)

		# sanity-check
		finally_overlapping = list(set(all_overlappping).intersection(set(test_classes)))
		print('Final overlapping: ', finally_overlapping)
		final_results['post_swap_trainval'] = al_classes
		final_results['post_swap_test'] = test_classes
		final_results['post_swap_overlapping'] = finally_overlapping

	

	plot_tsne_clusters(dataset, class_list, att_df, class_clusters_by_name, al_classes, args, new_dirs, final_flag = True)

	convert_to_mat(al_classes, image_folder, outfile, ratio=ratio, dataset=dataset, train_num=train_num)

	
	#NOTE: final_results['post_swap_test'] would contain only known unknown testclasses. To get all the testclasses finally present in .mat file, get them from both final_results['post_swap_test'] and final_results['common_unseen'] !!!!!

	pickle_file = new_dirs['split_info_dir'] + 'u_split' + str(args.sn) + '_' + 'split_info_' + dataset + '.pickle'
	pkl = open(pickle_file, 'wb')

	print('\n\nPlotting coverages.......')
	attcov.compute_cov(args, new_dirs, final_results, att_df)

	sectime = time.time() - prog_start
	print(time_in_dhms(sectime))
	final_results['runtime_in_dhms'] = time_in_dhms(sectime)
	pickle.dump(final_results, pkl)

	pkl.close()
	print("DONE")

	sys.stdout.close()
