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
import sklearn
import sys
import seaborn as sns
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image



def check_dirs(dataset, args, new_dirs):

	sem_cluster_dir = new_dirs['sem_cluster_dir'] + 'u_split' + str(args.sn) + '/'
	cov_dir = new_dirs['cov_dir'] + 'u_split' + str(args.sn) + '/'
	visual_dir = new_dirs['visual_dir'] + 'u_split' + str(args.sn) + '/'


	if not os.path.exists(sem_cluster_dir):
		os.mkdir(sem_cluster_dir)
	if not os.path.exists(cov_dir):
		os.mkdir(cov_dir)
	if not os.path.exists(visual_dir):
		os.mkdir(visual_dir)

	return sem_cluster_dir, cov_dir, visual_dir





def plot_silhouette_score(silhouette_avg_per_clustering, dataset, args, new_dirs):

	cluster_dir, _, _ = check_dirs(dataset, args, new_dirs)

	x = [silhouette_avg_per_clustering[i][0] for i in range(len(silhouette_avg_per_clustering))]
	y = [silhouette_avg_per_clustering[i][1] for i in range(len(silhouette_avg_per_clustering))]
	plt.figure(figsize=(15,10))
	plt.plot(x, y, 'm-', marker = 'o')
	plt.xlabel('Clusters', fontsize = 40)
	plt.ylabel('Silhouette score', fontsize = 40)
	plt.xticks(rotation = 90, fontsize = 20)
	plt.yticks(fontsize = 20)
	plotname = cluster_dir + 'u_split' + str(args.sn) + '_' + 'silhouette_scores_' + dataset +'.png'
	plt.savefig(plotname)


def plot_tsne_clusters(dataset, class_list, data, class_clusters_by_name, seed_classes, args, new_dirs, added_classes = None, currently_queried = None, query_iter = None, final_flag = False):

	# data = att_df matrix
	cluster_dir, _, _ = check_dirs(dataset, args, new_dirs)	

	# plot tsne of semantic embeddings and get cluster-wise colored results
	# data_transpose = data['original_att'].T   # just for clarity
	# num_classes = data_transpose.shape[0]
	# num_attributes = data_transpose.shape[1]

	num_classes = data.shape[0]
	num_attributes = data.shape[1]
	labels_to_plot = np.arange(num_classes)
	# labels_to_plot = np.array([17,  18,  19,  20, 0])
	# labels_to_plot = np.array([1,  2,  3,  4,  6,  8,  9, 10, 11, 12])
	# y_test_sem = np.arange(num_classes).reshape(-1, 1)
	z = []
	for c in class_clusters_by_name:
		z.extend(c[1])
	# z = [awa2['allclasses_names'][y_test_sem,0][i,0][0] for i in range(len(y_test_sem))]
	assert num_classes ==  len(z)
	CLASSES = np.concatenate(([], z))
	# print(CLASSES)
	id_to_class = {idx: class_label for idx, class_label in enumerate(CLASSES)}



	def tsne_plot_feats(f_feat, f_labels, path_save=cluster_dir):
		# import pdb; pdb.set_trace()
		tsne = TSNE(n_components=2, random_state=0, verbose=True)
		# syn_feature = f_feat
		# syn_label = f_labels
		# idx = np.where(np.isin(syn_label, labels_to_plot))[0]
		# idx = np.random.permutation(idx)[0:2000]
		# X_sub = syn_feature[idx]
		# y_sub = syn_label[idx]
		X_sub = f_feat
		y_sub = f_labels
		# targets = np.unique(y)

		colors = []
		seed_data = []
		p = 0
		for i, c in enumerate(class_clusters_by_name):
			col = '#%06X' % randint(0, 0xFFFFFF)
			for j in range(len(c[1])):
				if c[1][j] in seed_classes:
					colors.append(col)
					seed_data.append((p, c[1][j]))
				else:
					colors.append(col)
				p += 1

					
		if added_classes is not None:
			added_data = []
			currently_queried_data = []
			p = 0
			for i, c in enumerate(class_clusters_by_name):
				for j in range(len(c[1])):
					if c[1][j] in added_classes:
						added_data.append((p, c[1][j]))
					if c[1][j] in currently_queried:
						currently_queried_data.append(p)
					p += 1

			added_data = pd.DataFrame(added_data, columns=['addindex', 'addname'])
			addindex_list = added_data['addindex'].tolist()	
			added_cl = len(seed_classes) + 1
		# print(colors)
		# print(seed_index)
		# for i in range(len(labels_to_plot)):
		#     colors.append('#%06X' % randint(0, 0xFFFFFF))

		seed_data = pd.DataFrame(seed_data, columns=['seed_index', 'seed_name'])
		seed_index_list = seed_data['seed_index'].tolist()
		

		print(X_sub.shape, y_sub.shape, labels_to_plot.shape)

		X_2d = tsne.fit_transform(X_sub)
		# print(X_2d)
		# print(X_2d.shape)

		fig = plt.figure()
		cl = 1

		
		for i, c in zip(labels_to_plot, colors):
			if added_classes is None:
				if i in seed_index_list:
					
					ind = seed_index_list.index(i)
					plt.scatter(X_2d[y_sub == i, 0], X_2d[y_sub == i, 1], c=c, label = str(cl)+'. '+seed_data.iloc[ind][1])
					plt.annotate(str(cl), (X_2d[y_sub == i, 0], X_2d[y_sub == i, 1]))
					cl += 1

				else:
					plt.scatter(X_2d[y_sub == i, 0], X_2d[y_sub == i, 1], c=c)			

			else:
				if i in seed_index_list:
					
					ind = seed_index_list.index(i)
					plt.scatter(X_2d[y_sub == i, 0], X_2d[y_sub == i, 1], c=c, label = str(cl)+'. '+seed_data.iloc[ind][1])
					plt.annotate(str(cl), (X_2d[y_sub == i, 0], X_2d[y_sub == i, 1]))
					cl += 1

				elif i in addindex_list:
					ind = addindex_list.index(i)
					if i in currently_queried_data:
						plt.scatter(X_2d[y_sub == i, 0], X_2d[y_sub == i, 1], c=c, label = '*'+str(added_cl)+'. '+added_data.iloc[ind][1])
						plt.annotate(str(added_cl), (X_2d[y_sub == i, 0], X_2d[y_sub == i, 1]), color = 'red')
					else:
						plt.scatter(X_2d[y_sub == i, 0], X_2d[y_sub == i, 1], c=c, label = str(added_cl)+'. '+added_data.iloc[ind][1])
						plt.annotate(str(added_cl), (X_2d[y_sub == i, 0], X_2d[y_sub == i, 1]))
					added_cl += 1		

				else:
					plt.scatter(X_2d[y_sub == i, 0], X_2d[y_sub == i, 1], c=c)		



		plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
		# plt.show()
		if added_classes == None:
			if final_flag == True:
				figname = path_save + 'u_split' + str(args.sn) + '_' + dataset + '_tsne_train_classes.png'
			else:
				figname = path_save + 'u_split' + str(args.sn) + '_' + dataset + '_tsne_clusters.png'
		else:
			figname = path_save + 'u_split' + str(args.sn) + '_' + dataset + '_tsne_clusters_q' + str(query_iter) +'.png'

		fig.savefig(figname, bbox_inches = 'tight')
		print(f"saved {figname}")



	# indexes = []
	# for c in CLASSES:
	# 	for i in range(num_classes):
	# 		if c == data['allclasses_names'][i][0][0]:
	# 			indexes.append(i)
	# 			break


	# if dataset == 'SUN':
	# 	x_all_sem = (data['original_att'].T*100).reshape(-1, num_attributes)[indexes]
		
	# else:
	# 	x_all_sem = data['original_att'].T.reshape(-1, num_attributes)[indexes]

	indexes = []
	for c in CLASSES:
		for i in range(num_classes):
			if c == class_list[i]:
				indexes.append(i)
				break
	# got indexes in order of classes present in CLASSES. Now in x_all_sem, we can re-order the attribute matrix in this order to get apt semantic vectors

	np_data = data.to_numpy()   # converting dataframe object to numpy array
	x_all_sem = np_data.reshape(-1, num_attributes)[indexes]
	y_all_sem = np.arange(num_classes)
	tsne_plot_feats(x_all_sem, y_all_sem)



def plot_visual_mavs(args, new_dirs, ftrs, weibull_model, categories, c_or_q, query_iter, current_seed_names, current_candidate_names):

	# c_or_q defines whether candidate ftr ('c') or queried class feature (q)
	# currently visual plots for 'q' are not done - omission reason given in new_al_and_mat_openmax_randomTestSplits.py in function most_ambiguous_query()

	_, _, visual_dir = check_dirs(args.dataset, args, new_dirs)
	dir_min_candidates = visual_dir + 'min_candidates_u_split' + str(args.sn) + '/'
	dir_queried = visual_dir + 'queried_u_split' + str(args.sn) + '/'
	dir_activation_vecs = visual_dir + 'activation_vecs_u_split' + str(args.sn) + '/'

	if not os.path.exists(dir_min_candidates):
		os.mkdir(dir_min_candidates)
	if not os.path.exists(dir_queried):
		os.mkdir(dir_queried)
	if not os.path.exists(dir_activation_vecs):
		os.mkdir(dir_activation_vecs)

	# get array of MAVs
	mav_list = []
	for category_name in categories:
		mav_list.append(weibull_model[category_name]['mean_vec'][0])
	mav_array = np.asarray(mav_list)

	# get array of candidate/queried ftrs
	ftr_array = ftrs.reshape(ftrs.shape[0]*ftrs.shape[1], ftrs.shape[2])


	def tsne_visual_plot(f_feat, f_labels):

		tsne = TSNE(n_components=2, random_state=0, verbose=True)
		X_sub = f_feat
		y_sub = f_labels
		colors = []
		cdict = {0: 'blue', 1: 'red'}

		for i in range(2):
			# one color should for for mavs of seen classes - another for candidate/queried ftrs
			if i == 0:
				# designated color for seen mavs
				for j in range(mav_array.shape[0]):
					colors.append(i)
			else:
				# designated color for candidate/queried ftrs
				for j in range(ftr_array.shape[0]):
					colors.append(i)

		group = np.asarray(colors)
		X_2d = tsne.fit_transform(X_sub)
		fig = plt.figure()

		# for i, c in zip(labels_to_plot, colors):
		# 	if i < mav_array.shape[0]:
		# 		# all the seen mavs
		# 		plt.scatter(X_2d[y_sub == i, 0], X_2d[y_sub == i, 1], c=c, label = 'Current seen class MAVs')
		# 	else:
		# 		# all the candidate/queried ftrs
		# 		if c_or_q == 'c':
		# 			legend_label = 'Visual features of all candidates'
		# 		else:
		# 			legend_label = 'Visual features of queried candidates'

		# 		plt.scatter(X_2d[y_sub == i, 0], X_2d[y_sub == i, 1], c=c, label = legend_label)

		for g in np.unique(group):
			ix = np.where(group == g)
			if g == 0:
				lab = 'Current seen class MAVs'
			else:
				# all the candidate/queried ftrs
				if c_or_q == 'c':
					lab = 'Visual features of all candidates'
				else:
					lab = 'Visual features of queried candidates'

			plt.scatter(X_2d[y_sub[ix], 0], X_2d[y_sub[ix], 1], c = cdict[g], label = lab)


		plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
		
		if c_or_q == 'c':
			figname = dir_min_candidates + 'u_split' + str(args.sn) + '_' + args.dataset + '_all_candidates_q' + str(query_iter) + '.png'
		else:
			figname = dir_queried + 'u_split' + str(args.sn) + '_' + args.dataset + '_queried_candidates_q' + str(query_iter) + '.png'

		fig.savefig(figname, bbox_inches = 'tight')
		print(f"saved {figname}")


	x_visual_data = np.vstack((mav_array, ftr_array))
	names = current_seed_names + current_candidate_names
	labels_to_plot = np.arange((mav_array.shape[0] + ftr_array.shape[0]))
	tsne_visual_plot(x_visual_data, labels_to_plot)


	def activation_vec_plot(x_visual_data):

		fig2 = plt.figure()
		ax = sns.heatmap(x_visual_data, yticklabels = names, cmap="YlGnBu")
		for tick_label in ax.axes.get_yticklabels():
			if tick_label.get_text() in current_seed_names:
				tick_label.set_color("blue")
			else:
				tick_label.set_color("red")

		ax.tick_params(labelsize=2)
		figname = dir_activation_vecs + 'u_split' + str(args.sn) + '_' + args.dataset + '_actVecs_q' + str(query_iter) + '.pdf'
		fig2.savefig(figname, bbox_inches = 'tight')
		print(f"saved {figname}")

	# plot activation vectors - for SUN we wont be plotting for every query iter as there are too many classes - so labels wont be visible anymore
	if args.dataset == 'SUN':
		if query_iter <= 25:
			activation_vec_plot(x_visual_data)
	else:
		activation_vec_plot(x_visual_data)




# plot_att_covs is not feasible

def plot_att_covs(cov_dir, pklfile):

	# made proper changes here after integration with master file
	results = pickle.load(open(pklfile, 'rb'))

	x = [i for i in range(1, len(results['att'])+1)]
	plt.figure(figsize=(15,10))
	plt.plot(x, results['orig_coverage'], 'r-', label = 'Seen-class attribute coverage (existing split)', marker = 'o')
	plt.plot(x, results['our_coverage'], 'b-', label = 'Seen-class attribute coverage (proposed split)', marker = 'o')
	# in the above line, legend was mistakenly typed as ALIS split instead of proposed split - so all the output plots got a stale value. I have corrected it now here.
	plt.xlabel('Attribute', fontsize = 40)
	plt.ylabel('Coverage percentage', fontsize = 40)
	plt.xticks(rotation = 90, fontsize = 20)
	plt.yticks(fontsize = 20)
	plt.legend(loc="lower right", fontsize = 14)
	figname = cov_dir + 'att_coverage_final.png'
	plt.savefig(figname, bbox_inches = 'tight')


def plot_rarity_effects(args, effects_dict, model_paths):

	# make folders
	rarity_folder = '/workspace/arijit_pg/BTP2021/plots/' + args.dataset + '/rarity_effect_plots_lr' + str(args.al_lr) +'/'
	if not os.path.exists(rarity_folder):
		os.mkdir(rarity_folder)
	split_folder = rarity_folder + 'u_split' + str(args.sn) + '/'
	if not os.path.exists(split_folder):
		os.mkdir(split_folder)

	class_orders = {
	'rare_accs': 'rare_class_order', 
	'common_accs': 'common_class_order'
	}

	for m in model_paths.keys():
		for k, v in class_orders.items():
			es = effects_dict[m]['ES'][k]
			ps = effects_dict[m]['PS'][k]

			# just ensuring that in the x-axis class orders are same for both ES and PS
			assert effects_dict[m]['ES'][v] == effects_dict[m]['PS'][v]

			x = effects_dict[m]['PS'][v]
			# formatting: removing label numbers from the front of class names in case of CUB
			if args.dataset == 'CUB':
				# remove numbers from class names and replacing _ by space in CUB
				for i in range(len(x)):
					x[i] = x[i][4:]
					x[i] = x[i].replace('_', ' ')

			plt.figure(figsize=(15,10))
			plt.plot(x, es, 'r-', label = 'Existing Split (ES)', marker = 'o')
			plt.plot(x, ps, 'b-', label = 'Proposed Split (PS)', marker = 'o')
			plt.xlabel('Common unseen class', fontsize = 40)
			plt.ylabel('CZSL accuracy (in %)', fontsize = 40)
			plt.xticks(rotation = 90, fontsize = 20)
			plt.yticks(fontsize = 20)
			plt.legend(loc="upper right", fontsize = 14)
			figname = split_folder + 'u_split' + str(args.sn) + '_' + args.dataset + '_' + k + '_' + m +  '.pdf'
			plt.savefig(figname, bbox_inches = 'tight')


def plot_candidates(args, candidate_names_per_iter, query_iter, img_transform = None):

	# make folders
	candidate_img_folder = '/workspace/arijit_pg/BTP2021/plots/' + args.dataset + '/candidate_imgs_lr' + str(args.l_rate) + '_cq' + str(args.query_classes) + '/'
	if not os.path.exists(candidate_img_folder):
		os.mkdir(candidate_img_folder)
	split_folder = candidate_img_folder + 'u_split' + str(args.sn) + '/'
	if not os.path.exists(split_folder):
		os.mkdir(split_folder)

	imgs_this_iter = []
	for i in candidate_names_per_iter:
		image = Image.open(i).convert('RGB')
		image = img_transform(image)
		imgs_this_iter.append(image)

	tensor_candidate_imgs = torch.stack([i for i in imgs_this_iter], dim = 0)

	# make grid of imgs
	plt.figure(figsize=(25,10))
	plt.axis("off")
	figtitle = "Iteration " + str(query_iter)
	figname = split_folder + 'u_split' + str(args.sn) + '_' + args.dataset + '_' + 'vsm_iter' + str(query_iter) +  '.pdf'
	# plt.title(figtitle)
	plt.imshow(np.transpose(vutils.make_grid(tensor_candidate_imgs, padding=2, normalize=True, nrow = len(candidate_names_per_iter)),(1,2,0)))
	plt.savefig(figname, bbox_inches = 'tight')
	plt.clf()


	



	


















