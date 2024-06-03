import numpy as np
import pandas as pd
import scipy.io as sio
import os
import sys


def create_classfile(filename, data_path):

	# assuming that the semnatic matrix, class names, and attribute names - all the relevant matrices are stored in data_path

	att_mat_path = data_path + 'att_splits.mat'
	att_mat = sio.loadmat(att_mat_path)
	classes = att_mat['allclasses_names']

	with open(filename, 'w') as classfile:

		for i in range(classes.shape[0]):
			line = str(i+1) + '\t' + classes[i][0][0] + '\n'
			classfile.write(line)


def create_predfile(filename, data_path):

	# assuming that the semnatic matrix, class names, and attribute names - all the relevant matrices are stored in data_path

	att_mat_path = data_path + 'attributes.mat'
	att_mat = sio.loadmat(att_mat_path)
	preds = att_mat['attributes']

	with open(filename, 'w') as predfile:

		for i in range(preds.shape[0]):
			line = str(i+1) + '\t' + preds[i][0][0] + '\n'
			predfile.write(line)


def	get_semantic_matrix(data_path):

	att_mat_path = data_path + 'att_splits.mat'
	att_mat = sio.loadmat(att_mat_path)
	sem_mat =  att_mat['original_att'].T * 100  
	# since percentage values given in this dataset are divided by 100 already

	# np.savetxt(filename, sem_mat, fmt='%f')
	# using format of writing np array to txt file as float; otherwise default is saving with scientific notation
	return att_mat, sem_mat


def load_semantic_matrix(dataset):

	data_path = dataset + '/' + 'classes/'	
	
	if dataset == 'AWA2':

		filename = data_path + 'classes.txt'
		classfile = pd.read_csv(filename, sep = '\t',header = None)
		classfile.drop(classfile.columns[0], 1, inplace = True)
		print('Classes: {}'.format(classfile.shape))

		filename = data_path + 'predicates.txt'
		attNames = pd.read_csv(filename, sep = '\t',header = None)
		attNames.drop(attNames.columns[0], 1, inplace = True)
		print('Attributes: {}'.format(attNames.shape))

		filename = data_path + 'overlapping_awa2.txt'
		ol_data = pd.read_csv(filename, header = None)
		overlapping_classes = ol_data[0].tolist()


		num_classes = len(classfile)
		num_attributes = len(attNames)

		semantic_mat_path = data_path + 'att_splits.mat'
		data = sio.loadmat(semantic_mat_path)
		att_df = pd.DataFrame(data['original_att'].T, index = classfile[1].tolist(), columns= attNames[1].tolist())
		att_df[att_df < 0] = 0.0       # negative values not considered - made as 0.0
		print('Semantic space: {}'.format(att_df.shape))

		# get the test classes from Xian split to divide it into two splits later
		filename = data_path + 'testclasses.txt'
		ol_data = pd.read_csv(filename, header = None)
		given_testclasses = ol_data[0].tolist()


	elif dataset == 'CUB':

		filename = data_path + 'classes.txt'
		classfile = pd.read_csv(filename, sep = '\t',header = None)
		classfile.drop(classfile.columns[0], 1, inplace = True)
		classfile.head()
		print('Classes: {}'.format(classfile.shape))

		filename = data_path + 'attributes.txt'
		attNames = pd.read_csv(filename, sep = ' ',header = None)
		attNames.drop(attNames.columns[0], 1, inplace = True)
		print('Attributes: {}'.format(attNames.shape))

		filename = data_path + 'overlapping_cub.txt'
		ol_data = pd.read_csv(filename, header = None)
		overlapping_classes = ol_data[0].tolist()
		num_classes = len(classfile)
		num_attributes = len(attNames)

		semantic_mat_path = data_path + 'att_splits.mat'
		data = sio.loadmat(semantic_mat_path)
		att_df = pd.DataFrame(data['original_att'].T, index = classfile[1].tolist(), columns= attNames[1].tolist())
		att_df[att_df < 0] = 0.0        # negative values not considered - made as 0.0
		print('Semantic space: {}'.format(att_df.shape))


		# get the test classes from Xian split to divide it into two splits later
		filename = data_path + 'testclasses.txt'
		ol_data = pd.read_csv(filename, header = None)
		given_testclasses = ol_data[0].tolist()
		

	elif dataset == 'SUN':

		# getting formatted input
		#############################################################################################################
		filename = data_path + 'classes.txt'
		if not os.path.isfile(filename):
			# create file in proper format
			create_classfile(filename, data_path)

		classfile = pd.read_csv(filename, sep = '\t',header = None)
		classfile.drop(classfile.columns[0], 1, inplace = True)
		print('Classes: {}'.format(classfile.shape))


		filename = data_path + 'predicates.txt'
		if not os.path.isfile(filename):
			# create file in proper format
			create_predfile(filename, data_path)

		attNames = pd.read_csv(filename, sep = '\t',header = None)
		attNames.drop(attNames.columns[0], 1, inplace = True)
		print('Attributes: {}'.format(attNames.shape))

		filename = data_path + 'overlapping_sun.txt'
		ol_data = pd.read_csv(filename, header = None)
		overlapping_classes = ol_data[0].tolist()
		num_classes = len(classfile)
		num_attributes = len(attNames)

		# already transposed form in sem_mat
		data, sem_mat = get_semantic_matrix(data_path)

		att_df = pd.DataFrame(sem_mat)
		print('Semantic space: {}'.format(att_df.shape))
		# rename cols
		att_df.columns = attNames[1].tolist()      
		# rename rows
		att_df.index = classfile[1].tolist()

		att_df[att_df < 0] = 0.0   # negative values not considered - made as 0.0

		# get the test classes from Xian split to divide it into two splits later
		filename = data_path + 'testclasses.txt'
		ol_data = pd.read_csv(filename, header = None)
		given_testclasses = ol_data[0].tolist()


	return att_df, data, overlapping_classes, given_testclasses
