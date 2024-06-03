import numpy as np
import pandas as pd
import scipy.io as sio
import os
import sys
import argparse
import matplotlib.pyplot as plt
import pickle
from pprint import pprint


from load_semantics import *
from plots_with_av_heatmap import plot_rarity_effects

# model_paths = {
# 	'ALE': '/home/gdata/sandipan/BTP2021/new_zsl_models/ALE/CZSL',
# 	'ESZSL': '/home/gdata/sandipan/BTP2021/new_zsl_models/ESZSL/CZSL',
# 	'DEVISE': '/home/gdata/sandipan/BTP2021/new_zsl_models/DEVISE/CZSL',
# 	'SAE': '/home/gdata/sandipan/BTP2021/new_zsl_models/SAE/CZSL',
# 	'SJE': '/home/gdata/sandipan/BTP2021/new_zsl_models/SJE/CZSL',
# 	'LSRGAN': '/home/gdata/sandipan/BTP2021/new_zsl_models/LSRGAN/CZSL',
# 	'TFVAEGAN': '/home/gdata/sandipan/BTP2021/new_zsl_models/TFVAEGAN/tfvaegan-master/CZSL'
# }

model_paths = {
    #comment out CNZSL and TransZero if running for SUN dataset
	'ALE': '/workspace/arijit_pg/BTP2021/new_zsl_models/ALE/CZSL',
	'ESZSL': '/workspace/arijit_pg/BTP2021/new_zsl_models/ESZSL/CZSL',
	'DEVISE': '/workspace/arijit_pg/BTP2021/new_zsl_models/DEVISE/CZSL',
	'SAE': '/workspace/arijit_pg/BTP2021/new_zsl_models/SAE/CZSL',
	'SJE': '/workspace/arijit_pg/BTP2021/new_zsl_models/SJE/CZSL',
	'LSRGAN': '/workspace/arijit_pg/BTP2021/new_zsl_models/LSRGAN/CZSL',
	'TFVAEGAN': '/workspace/arijit_pg/BTP2021/new_zsl_models/TFVAEGAN/tfvaegan-master/CZSL',
	# 'CNZSL': '/workspace/arijit_pg/BTP2021/new_zsl_models/CNZSL/CZSL',
	'FREE': '/workspace/arijit_pg/BTP2021/new_zsl_models/FREE/CZSL',
 	# 'TransZero': '/workspace/arijit_pg/BTP2021/new_zsl_models/TransZero/CZSL',
	'MSDN': '/workspace/arijit_pg/BTP2021/new_zsl_models/MSDN/CZSL'
 
}

model_names_in_resfiles = {
	# as per the model names with which the result filenames start in CZSL folders
	'ALE': 'ale',
	'ESZSL': 'eszsl',
	'DEVISE': 'devise',
	'SAE': 'sae',
	'SJE': 'sje',
	'LSRGAN': 'clswgan',
	'TFVAEGAN': 'tfvaegan_czsl',
	# 'CNZSL':'cnzsl_czsl',
	'FREE':'free_czsl',
	# 'TransZero':'TransZero_czsl',
	'MSDN':'MSDN_czsl'
}

# classwise_accs_key_names = {
# 	# as per codes added by me for each model in its main python runner file
# 	'ALE': 'common_unseen_classwise',
# 	'ESZSL': 'common_unseen_classwise',
# 	'DEVISE': 'common_unseen_classwise',
# 	'SAE': 'common_unseen_classwise_F2S',
# 	'SJE': 'common_unseen_classwise',
# 	'LSRGAN': 'common_unseen_classwise',
# 	'TFVAEGAN': 'common_unseen_classwise'
# }

split_names = {
	'ES':'original', 
	'PS':'new_seed_final'
}


def get_paths():
	# ZSL folder name ends with lr value. For SUN it is 0.001 and for others it is 0.01. 
	split_paths = {k: v+'_al_lr'+str(args.al_lr)+'/u_split'+str(args.sn) for k, v in model_paths.items()}
	result_paths = {}
	resfiles = {'ES':{}, 'PS':{}}

	for m in model_paths.keys():
		top = split_paths[m]
		for root, dirs, files in os.walk(top, topdown=False):
			for file in files:
				if file.endswith(".pickle"):
					result_paths[m] = os.path.join(root)
					# print(result_paths[m])

		# segregate the result files for the current u_split
		for skey, sval in split_names.items():
			resfiles[skey][m] = result_paths[m] + '/' + model_names_in_resfiles[m] + '_' + args.dataset + '_' + sval + '_results.pickle'


	return split_paths, result_paths, resfiles


def get_domain_semantics():

	# get semantic matrices for trainval seen classes and common unseen classes from any resfile from any model - here we select ALE
	res = open(resfiles['PS']['ALE'], 'rb')
	test_res = pickle.load(res)
	common_unseen = test_res['zsl_common_unseen']
	att_df, data_complete_info, imagenet_overlapping_classes, given_testclasses = load_semantic_matrix(args.dataset)

	# for any run of our framework, the available experimental classes would be all except common unseen (say T). Hence, for identifying rare and common attributes, info should be extracted only from classes in T.
	common_unseen_att_df = att_df.loc[common_unseen]
	att_df.drop(common_unseen, axis = 0, inplace = True)
	res.close()
	return att_df, common_unseen_att_df, common_unseen


def get_rare_and_common_atts(x):

	r_ratio = float(args.r_ratio)
	c_ratio = float(args.c_ratio)

	# removing irrelevant and unremarkable attributes, if any (very low chance of getting any such attributes as we are considering the entire domain, except only few classes, i.e. the common unseen classes)
	non_zero_counts = np.count_nonzero(x, axis = 0)
	irrelevant_atts = x.columns[non_zero_counts == 0]
	print('\nIrrelevant_atts ({}) : {}'.format(len(irrelevant_atts), irrelevant_atts))
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
	unremarkable_atts = thresh_df.columns[non_zero_atts == 0]
	print('\nUnremarkable_atts ({}) : {}'.format(len(unremarkable_atts), unremarkable_atts))
	thresh_df = thresh_df.drop(columns= unremarkable_atts)

	# infer rare and common attributes from their frequencies in binary matrix
	occurences = thresh_df.sum()
	occurences.sort_values(inplace = True, ascending = True)

	# attribute will be rare if it occurs in less than r_ratio% of all domain classes
	rare_count = (occurences < round(thresh_df.shape[0] * r_ratio)).sum()
	rare_atts = occurences.index[:rare_count].tolist()
	# attribute will be common if it occurs in more than c_ratio% of all domain classes
	common_count = (occurences > round(thresh_df.shape[0] * c_ratio)).sum()
	common_atts = occurences.index[-common_count:].tolist() 
	# above line will mistakenly give all classes if common_count is zero. Tune r_ratio and c_ratio in such a way that this is avoided 
	print('\n\nRare ({}): {}'.format(rare_count, rare_atts))
	print('\n\nCommon ({}): {}'.format(common_count, common_atts))

	return rare_atts, common_atts, thresh_df
	

def get_classes_with_rare_and_common_atts(rare_atts, common_atts, common_unseen_att_df):

	# for common unseen classes, no thresholding and removal of irrelevant or unremarkable attributes is required - those were only for the classes incorporated in DiRaC-I. For easy processing, we consider any non-zero value in semantic matrix of common unseen classes as 1, otherwise a 0
	common_unseen_att_df[common_unseen_att_df > 0.0] = 1
	common_unseen_att_df = common_unseen_att_df.astype(int)

	# extracting rare info
	classes_with_rare = []   # contains all unique classes with at least one rare attribute
	rare_att_in_common_unseen = []    # contains all unique classes for each rare attribute
	for att in rare_atts:
		classes_found = common_unseen_att_df[common_unseen_att_df[att] == 1].index.tolist()
		rare_att_in_common_unseen.append(classes_found)

		for c in classes_found:
			if c not in classes_with_rare:
				classes_with_rare.append(c)

	# extracting common info
	classes_with_common = [] # contains all unique classes with at least one common attribute
	common_att_in_common_unseen = []  # contains all unique classes for each common attribute
	for att in common_atts:
		classes_found = common_unseen_att_df[common_unseen_att_df[att] == 1].index.tolist()
		common_att_in_common_unseen.append(classes_found)

		for c in classes_found:
			if c not in classes_with_common:
				classes_with_common.append(c)

	# creating dataframes
	rare_att_in_common_unseen = pd.DataFrame(rare_att_in_common_unseen, index = rare_atts)
	common_att_in_common_unseen = pd.DataFrame(common_att_in_common_unseen, index = common_atts)
	# print('\nResults for original unseen classes:')
	# print(rare_att_in_orig_test)
	# print(common_att_in_orig_test)
	print('\n\n')
	print('Classes with rare attributes ({}/{}): {}'.format(len(classes_with_rare), len(common_unseen_att_df.index), classes_with_rare))
	print('Classes with common attributes ({}/{}): {}'.format(len(classes_with_common), len(common_unseen_att_df.index), classes_with_common))

	return common_unseen_att_df, classes_with_rare, classes_with_common


def get_accuracies(classes_with_rare, classes_with_common, common_unseen, resfiles):

	print('\n\nImpact of incorporating rarity\n==============================\n')
	effects = {}
	for m in model_paths.keys():

		print('\nModel = ', m)
		effects[m] = {}
		if m == 'SAE':
			res_key = 'common_unseen_classwise_F2S'
		else:
			res_key = 'common_unseen_classwise'

		for split_name in resfiles.keys(): # ES or PS
			print('\nSplit name = ', split_name)
			pklfile = open(resfiles[split_name][m], 'rb')
			test_res = pickle.load(pklfile)
			classwise_accs_common_unseen = test_res[res_key]

			effects[m][split_name] = {}
			avg_rare = 0.0
			avg_common = 0.0
			rare_accs = []
			common_accs = []
			# sanity check
			rare_class_order = []
			common_class_order = []

			for c in common_unseen:
				a = classwise_accs_common_unseen[c] * 100 # converting to percentage values
				if c in classes_with_rare:
					rare_accs.append(a)
					rare_class_order.append(c)
					avg_rare += a
				if c in classes_with_common:
					common_accs.append(a)
					common_class_order.append(c)
					avg_common += a

			avg_rare = avg_rare/len(classes_with_rare)
			avg_common = avg_common/len(classes_with_common)

			print('\nAvg. accuracy for classes with rare attributes = ', avg_rare)
			print('\nAvg. accuracy for classes with common attributes = ', avg_common)

			effects[m][split_name]['avg_rare'] = avg_rare
			effects[m][split_name]['avg_common'] = avg_common
			effects[m][split_name]['rare_accs'] = rare_accs
			effects[m][split_name]['common_accs'] = common_accs
			effects[m][split_name]['rare_class_order'] = rare_class_order
			effects[m][split_name]['common_class_order'] = common_class_order

			pklfile.close()

	print('\n\n\nFinal rare effect results\n=========================\n')
	pprint(effects)

	return effects




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Observing the effect of rare attributes on learning ability of ZSL models")
	parser.add_argument('-d','--dataset', default = 'SUN', help = 'AWA2, SUN, CUB')
	parser.add_argument('-sn', '--sn', type = int, default = 1, help='random unknown unknown split number')
	parser.add_argument('-al_lr', '--al_lr', default=0.01, type=float, help='learning rate used during active learning')
	parser.add_argument('-rare', '--r_ratio', default=0.1, type=float, help='ratio below which attribute considered rare')
	parser.add_argument('-common', '--c_ratio', default=0.5, type=float, help='ratio above which attribute considered common')
	
	#added argument for better path parsing - change it as per you need
	parser.add_argument('--home_dir', default='/workspace/arijit_pg/BTP2021/', help='path to dataset')

	args = parser.parse_args()

	# make folder to store final reports
	res_folder = args.home_dir + args.dataset + '/rarity_reports_lr' + str(args.al_lr) + '/'
	if not os.path.exists(res_folder):
		os.mkdir(res_folder)
	result_filename = res_folder + 'u_split' + str(args.sn) + '_' + args.dataset + '_r' + str(args.r_ratio) + '_c' + str(args.c_ratio) +  '_rarity_reports.txt'
	sys.stdout = open(result_filename, 'w')

	# get the paths where results are stored
	split_paths, result_paths, resfiles = get_paths()
	pprint(resfiles)

	# get semantic matrix from which rare and common attributes are to be identified
	att_df, common_unseen_att_df, common_unseen = get_domain_semantics()
	num_domain_classes = att_df.shape[0]
	num_domain_atts = att_df.shape[1]
	print('\n\nDomain classes = {}, Domain attributes = {}'.format(num_domain_classes, num_domain_atts))
	rare_atts, common_atts, thresh_df = get_rare_and_common_atts(att_df)

	# obtain list of classes with rare and common attributes
	common_unseen_att_df_binary, classes_with_rare, classes_with_common = get_classes_with_rare_and_common_atts(rare_atts, common_atts, common_unseen_att_df)

	# extract classwise accuracies
	effects_dict = get_accuracies(classes_with_rare, classes_with_common, common_unseen, resfiles)
	storage_name = res_folder + 'u_split' + str(args.sn) + '_' + args.dataset + '_r' + str(args.r_ratio) + '_c' + str(args.c_ratio) +  '_rarity_effects.pickle'
	pfile = open(storage_name, 'wb')
	pickle.dump(effects_dict, pfile)
	pfile.close()

	# plot graphs
	plot_rarity_effects(args, effects_dict, model_paths)

	print('\n\nDone!')

	sys.stdout.close()