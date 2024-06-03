import numpy as np
import random
import os
import pickle
import pdb
import time
import sys
import scipy.spatial.distance as spd
import libmr

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


def softmax(x):

	# same result as formula-wise softmax for 1D array (x is already 1D since we used ravel() while calling softmax(x)). Check here for more details: https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
	# print('score shape: ',x.shape)
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()


def compute_openmax_prob(scores, scores_u):

	# Convert the scores in probability value using openmax
	prob_scores, prob_unknowns = [], []
	for s, su in zip(scores, scores_u):
		channel_scores = np.exp(s)
		channel_unknown = np.exp(np.sum(su))

		total_denom = np.sum(channel_scores) + channel_unknown
		prob_scores.append(channel_scores / total_denom)
		prob_unknowns.append(channel_unknown / total_denom)

	# Take channel mean
	scores = np.mean(prob_scores, axis=0)
	unknowns = np.mean(prob_unknowns, axis=0)
	modified_scores = scores.tolist() + [unknowns]
	return modified_scores


def calc_distance(query_score, mcv, eu_weight, distance_type='eucos'):
	
	# Compute the specified distance type between mean class vector and query image. Compute distance of each query (test) image with respective Mean Activation Vector. In the original paper, they considered a hybrid distance eucos which combines euclidean and cosine distance for bouding open space. Alternatively, other distances such as euclidean or cosine can also be used. 
	# mcv shape: (num_seen_classes, )
	# query_score = img_feature shape = (num_seen_classes, )
	
	if distance_type == 'eucos':
		query_distance = spd.euclidean(mcv, query_score) * eu_weight + spd.cosine(mcv, query_score)
	elif distance_type == 'euclidean':
		query_distance = spd.euclidean(mcv, query_score) * eu_weight
	elif distance_type == 'cosine':
		query_distance = spd.cosine(mcv, query_score)
	else:
		print("distance type not known: enter either of eucos, euclidean or cosine")
	return query_distance



def query_weibull(category_name, weibull_model, distance_type='eucos'):

	return [weibull_model[category_name]['mean_vec'], weibull_model[category_name]['distances_{}'.format(distance_type)], weibull_model[category_name]['weibull_model']]



def openmax(weibull_model, categories, input_score, alpha, eu_weight=0.005, distance_type='eucos'):
	
	"""Re-calibrate scores via OpenMax layer
	Output:
		openmax probability and softmax probability
	"""
	nb_classes = len(categories)
	# print('input ftrs: ', input_score.shape)
	ranked_list = input_score.argsort().ravel()[::-1][:alpha]
	# print('ranked_list: ', ranked_list.shape)
	alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
	omega = np.zeros(nb_classes)
	omega[ranked_list] = alpha_weights
	# omega = ranked_alpha, as per original code
	# each omega will store weights for only the top alpha scores

	# Now recalibrate each fc score for each class to include probability of unknown
	# scores represent openmax re-calibrated penultimate layer scores for each training class
	# scores_u represent openmax re-calibrated penultimate layer scores for unknown class
	scores, scores_u, all_dists = [], [], []
	for channel, input_score_channel in enumerate(input_score):
		score_channel, score_channel_u, dists_channel = [], [], []
		# print('For input_ftrs {}'.format(channel))
		for c, category_name in enumerate(categories):
			mav, dist, model = query_weibull(category_name, weibull_model, distance_type)
			# print('mav after query_weibull: ', mav.shape)
			channel_dist = calc_distance(input_score_channel, mav[channel], eu_weight, distance_type)
			# print('channel dist: ', channel_dist)
			# obtain w_score for the distance and compute probability of the distance being unknown wrt to mean training vector and channel(class) distances for category under consideration
			dists_channel.append(channel_dist)
			wscore = model[channel].w_score(channel_dist)
			modified_score = input_score_channel[c] * (1 - wscore * omega[c])
			score_channel.append(modified_score)
			score_channel_u.append(input_score_channel[c] - modified_score)

		scores.append(score_channel)
		scores_u.append(score_channel_u)
		all_dists.append(dists_channel)

	scores = np.asarray(scores)
	# print('scores: ', scores.shape)
	scores_u = np.asarray(scores_u)
	# print('scores_u: ', scores_u.shape)
	all_dists = np.asarray(all_dists)

	# Pass the recalibrated fc scores for the images into openmax 
	openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
	softmax_prob = softmax(np.array(input_score.ravel()))
	# print('opp: ', openmax_prob.shape)
	# print('sop: ', softmax_prob.shape)
	return openmax_prob, softmax_prob, all_dists



def fit_weibull(means, dists, categories, tailsize=20, distance_type='eucos'):
	"""
	Input:
		means (C, channel, C)
		dists (N_c, channel, C) * C
	Output:
		weibull_model : Perform EVT based analysis using tails of distances and save weibull model parameters for re-adjusting softmax scores
	"""
	# print('inside fit_weibull.......')
	# print('mavs: ', means.shape)
	# print('dists: ', len(dists))
	weibull_model = {}
	for mean, dist, category_name in zip(means, dists, categories):
		weibull_model[category_name] = {}
		weibull_model[category_name]['distances_{}'.format(distance_type)] = dist[distance_type]
		weibull_model[category_name]['mean_vec'] = mean
		weibull_model[category_name]['weibull_model'] = []
		for channel in range(mean.shape[0]):
			# each channel here corresponds to a seen category
			mr = libmr.MR()
			tailtofit = np.sort(dist[distance_type][channel, :])[-tailsize:]
			mr.fit_high(tailtofit, len(tailtofit))
			weibull_model[category_name]['weibull_model'].append(mr)

	# print('weibulls: ', len(weibull_model))
	# print('weibull dists: ', weibull_model[3]['distances_{}'.format(distance_type)])
	# print('weibull mean_vec: ', weibull_model[3]['mean_vec'])
	# print('weibull model: ', weibull_model[3]['weibull_model'])

	return weibull_model


def compute_channel_distances(mavs, features, eu_weight=0.005):
	"""
	Input:
		mavs (channel, C)
		features: (N, channel, C)
	Output:
		channel_distances: dict of distance distribution from MAV for each channel(seen class)
	"""
	# eu_weight has been set to the default of 1/200 = 0.005 as in original paper
	# here channel implies a particular seen class

	eucos_dists, eu_dists, cos_dists = [], [], []
	for channel, mcv in enumerate(mavs):  # Compute channel specific distances
		eu_dists.append([spd.euclidean(mcv, feat[channel]) for feat in features])
		cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])
		eucos_dists.append([spd.euclidean(mcv, feat[channel]) * eu_weight + spd.cosine(mcv, feat[channel]) for feat in features])

	return {'eucos': np.array(eucos_dists), 'cosine': np.array(cos_dists), 'euclidean': np.array(eu_dists)}



def get_mavs_and_dists(model, device, tr_labeled_loader, num_seen_classes, current_a_labels):

	# store the penultimate layer features of each sample of each class
	features = [[] for _ in range(num_seen_classes)]  
	imgs_per_seen_class = {l:0 for l in current_a_labels} 
	# only training img count per class (from the 90% split) 

	with torch.no_grad():
		for batch in tr_labeled_loader:

			inputs, targets, _, _, a_targets = batch
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = model(inputs)
			# probabilities = F.softmax(outputs, dim=1)

			# output vector will have shape = number of seen classes = C

			for f, t, a in zip(outputs, targets, a_targets):
				# print(f"torch.argmax(score) is {torch.argmax(score)}, t is {t}")
				# print('f:{} t: {} '.format(torch.argmax(f), t))
				if torch.argmax(f) == t:
				    features[t].append(f.unsqueeze(dim=0).unsqueeze(dim=0))
				imgs_per_seen_class[a.item()] += 1

	# print('imgs_per_seen_class: ', imgs_per_seen_class)
	# print('tr ftrs before concat: ', len(features))
	for i in range(len(features)):
		if len(features[i]) == 0:
			print('at {}: {}'.format(i, len(features[i])), end = '\t')
			print(imgs_per_seen_class)
			
	features = [torch.cat(x).cpu().numpy() for x in features]  # (N_c, 1, C) * C
	# for i in range(len(features)):
	# 	print('after cat at {}: {}'.format(i, features[i].shape))
	# print('each tr ftr after concat: ', features[0].shape)
	# print('tr ftrs after concat:', len(features))
	mavs = np.array([np.mean(x, axis=0) for x in features])  # (C, 1, C)
	# print('mavs:', mavs.shape)
	dists = [compute_channel_distances(mcv, ftrs) for mcv, ftrs in zip(mavs, features)]
	# print('dists: ', len(dists))
	return features, mavs, dists, imgs_per_seen_class



