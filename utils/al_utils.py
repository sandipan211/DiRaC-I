import numpy as np
import pandas as pd
import scipy.io as sio
import math
from sklearn.cluster import AgglomerativeClustering 
from scipy.spatial import distance_matrix
from sklearn.metrics import silhouette_score, silhouette_samples
from random import randint
import os
import time
import sys


def kcenters(att_df, seed_classes, k = 2, tolerance = 0.9):

	unseen_classes = list(set(att_df.index) - set(seed_classes))
	seen_matrix = att_df.loc[seed_classes]
	unseen_matrix = att_df.loc[unseen_classes]

	kcent_dist = pd.DataFrame(distance_matrix(unseen_matrix.values, seen_matrix.values), index = unseen_matrix.index, columns= seen_matrix.index)
	min_kcent_dist = kcent_dist.min(axis = 1)
	max_min_kcent_dist = tolerance * min_kcent_dist.max()
	print(min_kcent_dist[min_kcent_dist > max_min_kcent_dist])


def find_CRs(train_loader, train_labels):

	def get_medoid(features):

		dist_mat = distance_matrix(features, features)
		medoid_index = np.argmin(np.sum(dist_mat, axis=1))
		return features[medoid_index]



	# # loads entire input data at once - can be memory-consuming
	# ftrs, targets, _ = next(iter(train_loader))
	start_time = time.time()

	obtd_CRs = {}
	imgs_per_class = {}
	ftr_lists = [[] for x in range(len(train_labels))]

	for batch in train_loader:
		_, targets, _, ftrs = batch
		ftrs = ftrs.cpu().data.numpy()
		targets = targets.cpu().data.numpy()

		for lab_idx, label in enumerate(train_labels):
			ftr_indices = np.where(targets == label)[0]
			if len(ftr_indices) > 0:	# if at least one match
				obtd_ftrs = ftrs[ftr_indices]
				ftr_lists[lab_idx].append(obtd_ftrs)



	for lab_idx, label in enumerate(train_labels):
		print("label index: {} label: {}".format(lab_idx, label))
		ftrs_array = np.concatenate(ftr_lists[lab_idx], axis = 0)
		ftrs_array = np.reshape(ftrs_array, (ftrs_array.shape[0]*ftrs_array.shape[1], ftrs_array.shape[2]))
		print("array shape: ", ftrs_array.shape)
		obtd_CRs[label] = get_medoid(ftrs_array)
		imgs_per_class[label] = ftrs_array.shape[0]

	print('Time for obtaining {} CRs: {}'.format(len(obtd_CRs), time.time() - start_time))
	return obtd_CRs, imgs_per_class


def find_kNearest(features, all_CRs, k = 3):

	features = np.reshape(features, (features.shape[0]*features.shape[1], features.shape[2]))
	CR_labels = list(all_CRs.keys())
	CR_idx_to_label = {k:v for k, v in enumerate(CR_labels)}
	CR_list = [all_CRs[i] for i in CR_labels]
	CR_array = np.stack(CR_list, axis = 0)

	# print("features.shape ",features.shape)
	# print("CR_array.shape", CR_array.shape)
	dist_mat = distance_matrix(features, CR_array)
	knns = np.argpartition(dist_mat, range(k), axis=1)[:, :k]
	# np.argpartition will give only the top k smallest elements of each row - after k smallest elements it doesn't care to sort the others
	label_wise_knns = np.copy(knns)
	for idx, label in CR_idx_to_label.items():
		label_wise_knns[knns == idx] = label

	# label_wise_knns now contains the actual labels as the array values for KNNs of each image
	# print("label wise knn shape", label_wise_knns.shape)
	return label_wise_knns


def get_importance(att_df, imgs_per_seen_class, candidate_names, label_to_class, class_labels):

	total_seen_imgs = sum(imgs_per_seen_class.values())
	seen_classes = [label_to_class[k] for k, v in imgs_per_seen_class.items()]
	seen_imgs = [v/100 for k, v in imgs_per_seen_class.items()]
	# since att_df has percentage values in range[0, 100] so making it to [0,1]
	imgs_with_atts = att_df.loc[seen_classes].mul(seen_imgs, axis = 0)
	# print('before to int: ', imgs_with_atts)
	imgs_with_atts = imgs_with_atts.astype(int)

	# after making all columns as int, in the initial stages it may happen that some attributes are not covered by any image - so entire column for that attribute is zero. Weights should not be computed for them - so we are removing those attributes
	non_zero_atts = np.count_nonzero(imgs_with_atts, axis = 0)
	unqualified_atts = imgs_with_atts.columns[non_zero_atts == 0]
	print('\nUnqualified_atts : {}'.format(unqualified_atts.shape))

	imgs_with_atts = imgs_with_atts.drop(columns= unqualified_atts)

	print(imgs_with_atts.shape)
	att_ratios = imgs_with_atts.sum()/total_seen_imgs
	print('att ratios: ', att_ratios.shape)
	att_weights = np.log(1/att_ratios)

	candidates = att_df.loc[candidate_names]
	candidates = candidates.drop(columns = unqualified_atts)
	dominance_per_class = candidates.dot(att_weights)
	candidate_class_imps = {class_labels[dominance_per_class.index.values[i]]:dominance_per_class[i] for i in range(len(dominance_per_class))}

	print('candidate_names: ', candidate_names)
	# print('candidate_labels: ')
	# for i in candidate_names:
	# 	print(class_labels[i])

	# print('scores: ', candidate_class_imps)

	return candidate_class_imps, att_ratios



 