import pickle as p
import pandas as pd
import argparse
import re
import os
import sys

parser = argparse.ArgumentParser(description="imgcount")

parser.add_argument('-data', '--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='AWA2', type=str)
parser.add_argument('-model', '--model', help='choose between ALE, SAE, SJE, DEVISE, ESZSL, LSRGAN, TFVAEGAN', default='ALE', type=str)
parser.add_argument('-res_model', '--res_model', help='choose between ale, sae, sje, devise, eszsl, clswgan, tfvaegan_czsl', default='ale', type=str)
parser.add_argument('-sn', '--sn', default=1, type=int, help='split number')
parser.add_argument('-al_lr', '--al_lr', default=0.01, type=float, help='learning rate used during active learning')
parser.add_argument('-al_seed', '--al_seed', help = 'new_seed_final, random, original', default = 'new_seed_final', type =str)



def get_img_count(rf, storage, image_folder):

	###### getting number of images of each class - change it as per folder structure of each dataset ########
	filename = rf + args.res_model + '_' + args.dataset + '_' + args.al_seed + '_' + 'results.pickle' 
	pklfile = open(filename, 'rb')
	data = p.load(pklfile)

	resfile = storage + 'u_split' + str(args.sn) + '_' + 'image_counts_' + args.dataset + '.txt'
	sys.stdout = open(resfile, 'w')

	set_type = ['zsl_train', 'zsl_val', 'zsl_test', 'zsl_common_unseen']

	print('From train and val sets, only 20 percent data was shifted to test seen set. But in this file, the complete set of images are shown for train, val, test and test common unseen\n=============================================')
	#explanation: let k = int(ratio * len(temp1_loc)) as used in the main AL code while making .mat files. Then during obtaining train and val, we take data_loc[0:k], i.e total k+1 images from set data_loc

	for t in set_type:

		img_list = []
		for c in data[t]:
			class_path = image_folder + c +'/'
			image_names = [f for f in os.listdir(class_path) if re.search(r'.*\.(jpg|jpeg|png)$', f)]
			img_list.append(len(image_names))

		imgs_per_class = pd.Series(data = img_list, index = data[t], name = 'imgs_per_class')

		print(t)
		print('---------------------------')
		print(imgs_per_class)
		print('\nTotal images from {} classes = {}'.format(len(data[t]), imgs_per_class.sum()))


		print('\n\n')

	###########################################################################################################


if __name__ == '__main__':
	
	args = parser.parse_args()

	#pickle files containing train, val, test classes are given in every model of new_zsl_models folder. Enter name of any model from which you want to pick the pickle file (prefereably formatted for ale, sae, sje, devise and eszsl). By default I am taking from ALE

	res_folder = '/home/gdata/sandipan/BTP2021/new_zsl_models/' + args.model + '/CZSL_al_lr' + str(args.al_lr) + '/u_split' + str(args.sn) + '/'

	storage = '/home/gdata/sandipan/BTP2021/' + args.dataset + '/split_info_lr' + str(args.al_lr) + '/'

	image_folder = '/home/gdata/sandipan/BTP2021/' + args.dataset + '/Data/'

	pd.set_option("display.max_rows", None, "display.max_columns", None) # display all rows and columns with pandas

	get_img_count(res_folder, storage, image_folder) 

	sys.stdout.close()

