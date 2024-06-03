import os
import sys


def create_reports_dir(opt):
	'''
	creates a folder in the dataset-named directory which contains report files for entire AL mechanism for different random splits of (unknown unknown) classes

	'''
	save_directory_path = opt.dataset + '/' + 'al_reports_lr' + str(opt.l_rate) + '_cq' + str(opt.query_classes) +'/'
	if not os.path.exists(save_directory_path):
		os.mkdir(save_directory_path)

	return save_directory_path


def create_clusters_dir(opt):
	'''
	creates a folder in the dataset-named directory which contains clustering results for different random splits of (unknown unknown) classes

	'''
	save_directory_path = opt.dataset + '/' + 'clustering_results_lr' + str(opt.l_rate)  + '_cq' + str(opt.query_classes) +'/'
	if not os.path.exists(save_directory_path):
		os.mkdir(save_directory_path)

	return save_directory_path


def create_plots_dir(opt):
	'''
	creates a folder in the plots/ directory which contains semantic plots, visual mav plots and coverage plots for different random splits of (unknown unknown) classes

	'''
	plots_dir = '/home/gdata/sandipan/BTP2021/plots/'
	if not os.path.exists(plots_dir):
		os.makedirs(plots_dir)

	# make data folder inside plots
	data_folder = plots_dir + opt.dataset + '/'
	if not os.path.exists(data_folder):
		os.makedirs(data_folder)

	sem_cluster_dir = plots_dir + opt.dataset +'/' + 'semantic_cluster_plots_lr'+ str(opt.l_rate) + '_cq' + str(opt.query_classes) +'/'
	cov_dir = plots_dir + opt.dataset +'/' + 'coverage_plots_lr' + str(opt.l_rate) + '_cq' + str(opt.query_classes) +'/'
	visual_dir = plots_dir + opt.dataset +'/' + 'visual_mav_plots_lr' + str(opt.l_rate)  + '_cq' + str(opt.query_classes) +'/'

	if not os.path.exists(sem_cluster_dir):
		os.mkdir(sem_cluster_dir)
	if not os.path.exists(cov_dir):
		os.mkdir(cov_dir)
	if not os.path.exists(visual_dir):
		os.mkdir(visual_dir)

	return sem_cluster_dir, cov_dir, visual_dir


def create_split_info_dir(opt):
	'''
	creates a folder in the dataset-named directory which contains split info files for different random splits of (unknown unknown) classes

	'''
	save_directory_path = opt.dataset + '/' + 'split_info_lr' + str(opt.l_rate)  + '_cq' + str(opt.query_classes) +'/'
	if not os.path.exists(save_directory_path):
		os.mkdir(save_directory_path)

	return save_directory_path

def make_dirs(opt):

	new_dirs = {}

	d = create_reports_dir(opt)
	new_dirs['reports_dir'] = d


	d = create_clusters_dir(opt)
	new_dirs['clusters_dir'] = d

	cl, cov, v = create_plots_dir(opt)
	new_dirs['sem_cluster_dir'] = cl
	new_dirs['cov_dir'] = cov
	new_dirs['visual_dir'] = v

	d = create_split_info_dir(opt)
	new_dirs['split_info_dir'] = d


	return new_dirs
