import numpy as np
import pandas as pd
import scipy.io as sio
import math
from sklearn.cluster import AgglomerativeClustering 
from scipy.spatial import distance_matrix
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pdb
from sklearn.manifold import TSNE
from random import randint
import os
import pickle
import sys


def cluster_classes(att_df, linkage, num_classes):

	X = att_df.iloc[:].values
	vary_clusters = [i for i in range(2, num_classes)]

	clustered_classes = []
	silhouette_avg_per_clustering = []
	
	for n_clusters in vary_clusters:

		clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)  
		clusterer.fit_predict(X)
		cluster_labels = clusterer.labels_
		clustered_classes.append((n_clusters, cluster_labels))
		
		silhouette_avg = silhouette_score(X, cluster_labels)
		silhouette_avg_per_clustering.append((n_clusters, silhouette_avg))
		

	return clustered_classes, silhouette_avg_per_clustering


def get_class_clusters(min_seeds, att_df, clustered_classes, silhouette_avg_per_clustering):

	# returns the classes as per their clusters. Number of clusters selected corresponds to the one that attains highest silhouette score, where num_of_clusters = max(min_seeds, num_classes-1) 
	class_clusters_by_name = []
	highest_index = min_seeds - 2

	for i in range(min_seeds - 2, len(silhouette_avg_per_clustering)):
		if silhouette_avg_per_clustering[i][1] > silhouette_avg_per_clustering[highest_index][1]:
			highest_index = i


	for i in range(len(np.unique(clustered_classes[highest_index][1]))):
		indices = [k for k, x in enumerate(clustered_classes[highest_index][1]) if x == i]
		classes = att_df.index[indices].tolist()
		class_clusters_by_name.append((i+1, classes))

	return class_clusters_by_name


def get_seeds(att_df, class_clusters_by_name):

	# picks the best seed from each cluster according to  attribute scoring and returns the list of seeds
	# medoid distance was earlier considered, but not anymore
	
	seed_classes = []
	cluster_att_info = {}

	for cluster_num in range(len(class_clusters_by_name)):

		print('\nCluster number {} \t Classes: {}'.format(cluster_num+1, class_clusters_by_name[cluster_num][1]))

		if len(class_clusters_by_name[cluster_num][1]) > 1:
			cluster_number = cluster_num+1
			cluster_att_info[cluster_number] = {}

			# reject irrelevant attributes
			x = att_df.loc[class_clusters_by_name[cluster_num][1]]
			# print(x)
			non_zero_counts = np.count_nonzero(x, axis = 0)
			irrelevant_atts = x.columns[non_zero_counts == 0]
			print('\nIrrelevant_atts : {}'.format(irrelevant_atts))
			x = x.drop(columns= irrelevant_atts)
			non_zero_counts = non_zero_counts[non_zero_counts != 0]

			# calculate thershold for each remaining attribute
			clipped_semantic_mean = x.sum()
			clipped_semantic_mean = clipped_semantic_mean / non_zero_counts	# mean of only nonzero values of each attribute
			# print(clipped_semantic_mean)

			# get binary semantic matrix
			thresh_df = (x - clipped_semantic_mean).clip(lower=0)
			thresh_df[thresh_df > 0] = 1
			thresh_df = thresh_df.astype(int)
			non_zero_atts = np.count_nonzero(thresh_df, axis = 0)
			unqualified_atts = thresh_df.columns[non_zero_atts == 0]
			print('\nUnqualified_atts : {}'.format(unqualified_atts))

			thresh_df = thresh_df.drop(columns= unqualified_atts)

			x = x.drop(columns = unqualified_atts)
			# x has original semantic matrix for classes in recent cluster with relevant and qualified atts only
			# dist_mat =  pd.DataFrame(distance_matrix(x.values, x.values), index=x.index, columns=x.index)
			# dist_mat

			cooc_df = thresh_df.T.dot(thresh_df)
			att_occ_in_cluster = pd.Series(np.diag(cooc_df), index = cooc_df.index)

			# save att frequencies
			cluster_att_info[cluster_number]['freq'] = att_occ_in_cluster

			att_occ_in_cluster = 1/ att_occ_in_cluster.div(x.shape[0])  
			# if occurrence of attribute is occ/num_classes = x
			att_weights = np.log(att_occ_in_cluster)
			# then weights are given by ln(1/x). so rarer the attribute is, more will be the weight
			# att_weights

			# save att weights
			cluster_att_info[cluster_number]['theta'] = att_weights

			semantics_for_att_imp = np.multiply(thresh_df, x)
			final_weight_per_class = semantics_for_att_imp.dot(att_weights)
			print(final_weight_per_class)



			seed_classes.append(final_weight_per_class.index[final_weight_per_class.argmax()])
			print('\n\nSeed class added : {}'.format(final_weight_per_class.index[final_weight_per_class.argmax()]))
			# seed

		elif len(class_clusters_by_name[cluster_num][1]) == 1:
			seed_classes = seed_classes + class_clusters_by_name[cluster_num][1]
			print('\n\nSeed class added : {}'.format(class_clusters_by_name[cluster_num][1][0]))


	return seed_classes, cluster_att_info





