"""
Calls code from everywhere else and gets things done.
"""
import os, sys, pickle
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics

from data_utils import tsio
from data_utils import settings
from data_utils import transform
from data_utils import feat_extract as fe
from models import dom_adapt

def simple_per_person():
	"""
	Learn on one person and test on the same person. Does this for all the 
	users. Also uses 5 different random seeds for each user and averages 
	performance across the 5 seeds.
	"""
	results = dict()
	for usr_str in settings.usr_li:
		# Read and structure data appropriately.
		user_data, _ = transform.read_feats(source_li=[usr_str],
			target_li=None)
		print('User: {:s}; User data: {}'.format(usr_str, user_data.shape))
		user_precision = list()
		user_recall = list()
		user_accuracy = list()
		# Find metrics for different random seeds and average across these
		# splits.
		for random_seed in settings.random_seeds:
			# Create test, train split.
			train, test = model_selection.train_test_split(user_data, 
				test_size=0.2, random_state=random_seed)
			train_y = train['label']
			test_y = test['label']
			train_x = train.drop('label', axis=1)
			test_x = test.drop('label', axis=1)

			# Train model.
			rf_clf = ensemble.RandomForestClassifier(n_estimators=10, 
				criterion='entropy', max_features='sqrt', random_state=random_seed)
			rf_clf.fit(train_x, train_y)

			# Test on test data.
			preds = rf_clf.predict(test_x)

			# Evaluate the results.
			user_precision.append(metrics.precision_score(test_y, preds, 
				labels=range(1,len(settings.act_li)+1,1), average='weighted'))
			user_recall.append(metrics.recall_score(test_y, preds, 
				labels=range(1,len(settings.act_li)+1,1), average='weighted'))
			user_accuracy.append(metrics.accuracy_score(test_y, preds, 
				normalize=True))
		pre = np.mean(user_precision)
		rec = np.mean(user_recall)
		acu = np.mean(user_accuracy)
		print('precision: {:4f}, recall: {:4f} accuracy: {:4f}'.
			format(pre, rec, acu))
		sys.stdout.flush()
		results[usr_str]= np.array([pre, rec, acu])
	# Compute average over all users.
	overall_res = np.mean(np.array(results.values()), axis=0)
	print('Averaged across all users\nprecision: {:4f}, recall: {:4f},'\
		' accuracy: {:4f}'.format(overall_res[0], overall_res[1], 
		overall_res[2]))

def simple_inter_person():
	"""
	Learn on multiple people and test on the same set of people. Does this for 
	four groups which consist of different kinds of people. Also uses 5 
	different random seeds for each group and averages performance across 
	the runs.
	"""
	results = dict()
	for group_li in settings.groups:
		group_str = ' '.join(group_li)
		group_data, _ = transform.read_feats(source_li=group_li,
				target_li=None)
		print('Group: {:s}; Group data: {}'.format(group_str, group_data.shape))
		group_precision = list()
		group_recall = list()
		group_accuracy = list()
		for random_seed in settings.random_seeds:
			# Create test, train split.
			train, test = model_selection.train_test_split(group_data,
				test_size=0.2, random_state=random_seed)
			train_y = train['label']
			test_y = test['label']
			train_x = train.drop('label', axis=1)
			test_x = test.drop('label', axis=1)

			# Train model.
			rf_clf = ensemble.RandomForestClassifier(n_estimators=10,
				criterion='entropy', max_features='sqrt', 
				random_state=random_seed)
			rf_clf.fit(train_x, train_y)

			# Test on test data.
			preds = rf_clf.predict(test_x)

			# Evaluate the results.
			group_precision.append(metrics.precision_score(test_y, preds, 
				labels=range(1,len(settings.act_li)+1,1), average='weighted'))
			group_recall.append(metrics.recall_score(test_y, preds, 
				labels=range(1,len(settings.act_li)+1,1), average='weighted'))
			group_accuracy.append(metrics.accuracy_score(test_y, preds, 
				normalize=True))
		pre = np.mean(group_precision)
		rec = np.mean(group_recall)
		acu = np.mean(group_accuracy)
		print('precision: {:4f}, recall: {:4f} accuracy: {:4f}'.
			format(pre, rec, acu))
		sys.stdout.flush()
		results[group_str]= np.array([pre, rec, acu])
	# Compute average over all groups.
	overall_res = np.mean(np.array(results.values()), axis=0)
	print('Averaged across all groups\nprecision: {:4f}, recall: {:4f},'\
		' accuracy: {:4f}'.format(overall_res[0], overall_res[1], 
		overall_res[2]))

def plain_transfer_person(allow_peeking, group):
	"""
	Learn on some people and test on entirely different person or people.
	Setting allow_peeking to True allows the source subset to use 10% of the
	target data. This is for experimenting with the material presented in the 
	TPDA paper.
	"""
	# Pick group of people like one another.
	if group == 'similar':
		tl_groups = settings.tl_groups_similar
	elif group == 'diff':
		tl_groups = settings.tl_groups_diff

	print('Allow peeking: {}; Group: {:s}'.format(allow_peeking, group))
	results = dict()
	for tl_group in tl_groups:
		group_str = ' '.join([val for sublist in tl_group.values() for val in sublist])
		source_data, target_data = transform.read_feats(
			source_li=tl_group['source'], target_li=tl_group['target'])
		tl_group_precision = list()
		tl_group_recall = list()
		tl_group_accuracy = list()
		for random_seed in settings.random_seeds:
			# Let the training see some of the target data.
			if allow_peeking:
				tar_train_subset, tar_test_subset = model_selection.train_test_split(
					target_data, test_size=0.9, random_state=random_seed)
				source_data = pd.concat([source_data, tar_train_subset], axis=0)
				target_data = tar_test_subset
			if random_seed == settings.random_seeds[0]:
				print('TL Group: {}\nSource data: {} Target data: {}'.\
					format(group_str, source_data.shape, target_data.shape))

			# Create test, train split.
			source_y = source_data['label']
			target_y = target_data['label']
			source_x = source_data.drop('label', axis=1)
			target_x = target_data.drop('label', axis=1)
			
			# Train model.
			rf_clf = ensemble.RandomForestClassifier(n_estimators=10, 
				criterion='entropy', max_features='sqrt', random_state=random_seed)
			rf_clf.fit(source_x, source_y)

			# Test on test data.
			preds = rf_clf.predict(target_x)

			# Evaluate the results.
			tl_group_precision.append(metrics.precision_score(target_y, preds, 
				labels=range(1,len(settings.act_li)+1,1), average='weighted'))
			tl_group_recall.append(metrics.recall_score(target_y, preds, 
				labels=range(1,len(settings.act_li)+1,1), average='weighted'))
			tl_group_accuracy.append(metrics.accuracy_score(target_y, preds, 
				normalize=True))
		pre = np.mean(tl_group_precision)
		rec = np.mean(tl_group_recall)
		acu = np.mean(tl_group_accuracy)
		print('precision: {:4f}, recall: {:4f} accuracy: {:4f}'.
			format(pre, rec, acu))
		sys.stdout.flush()
		results[group_str]= np.array([pre, rec, acu])
	# Compute average over all tl groups.
	overall_res = np.mean(np.array(results.values()), axis=0)
	print('Averaged across all groups\nprecision: {:4f}, recall: {:4f},'\
		' accuracy: {:4f}'.format(overall_res[0], overall_res[1], 
		overall_res[2]))

def smart_transfer_person(method, group):
	"""
	Train a group of people and test on a completely different set of people.
	Allows for the application of one of two methods:
		- Topology preserving domain adaptation. (TPDA)
		- Modified TPDA.
	"""
	# Pick group of people like one another.
	if group == 'similar':
		tl_groups = settings.tl_groups_similar
	elif group == 'diff':
		tl_groups = settings.tl_groups_diff
	
	print('Using method: {:s}; Group: {:s}'.format(method, group))
	results = dict()
	for tl_group in tl_groups:
		group_str = ' '.join([val for sublist in tl_group.values() for val in sublist])
		source_data, target_data = transform.read_feats(
				source_li=tl_group['source'], target_li=tl_group['target'])
		tl_group_precision = list()
		tl_group_recall = list()
		tl_group_accuracy = list()
		for random_seed in settings.random_seeds:
			# Form small portion of target data for training.
			tar_data_tr, tar_data_te = model_selection.train_test_split(
				target_data, test_size=0.9, random_state=random_seed)
			target_y = tar_data_te['label']
			target_x = tar_data_te.drop('label', axis=1)
			if random_seed == settings.random_seeds[0]:
				print('TL Group: {}\nSource data: {}, Target train:{},'
					' Target test: {}'.format(group_str, source_data.shape, 
					tar_data_tr.shape, tar_data_te.shape))
			
			# Train the TPDA or mTPDA model.
			if method is 'tpda':
				tpda = dom_adapt.TopologyPreservingDA(n_iter=5, isomap_comp=15, 
					nn_neighbors=10, random_state=random_seed, clf_str='rf', 
						verbose=False)
				tpda.fit(tar_data_tr, source_data)
				preds = tpda.predict(target_x)
			elif method is 'mtpda':
				# mtpda doesn't look at labels.
				tar_data_tr = tar_data_tr.drop('label', axis=1)
				tpda = dom_adapt.modTopologyPreservingDA(n_iter=1, isomap_comp=15, 
					nn_neighbors=10, random_state=random_seed, clf_str='rf', 
					verbose=False)
				tpda.fit(tar_data_tr, source_data)
				preds = tpda.predict(target_x)

			# Evaluate the results.
			tl_group_precision.append(metrics.precision_score(target_y, preds, 
				labels=range(1,len(settings.act_li)+1,1), average='weighted'))
			tl_group_recall.append(metrics.recall_score(target_y, preds, 
				labels=range(1,len(settings.act_li)+1,1), average='weighted'))
			tl_group_accuracy.append(metrics.accuracy_score(target_y, preds, 
				normalize=True))
		pre = np.mean(tl_group_precision)
		rec = np.mean(tl_group_recall)
		acu = np.mean(tl_group_accuracy)
		print('precision: {:4f}, recall: {:4f} accuracy: {:4f}'.
			format(pre, rec, acu))
		sys.stdout.flush()
		results[group_str]= np.array([pre, rec, acu])
	# Compute average over all tl groups.
	overall_res = np.mean(np.array(results.values()), axis=0)
	print('Averaged across all groups\nprecision: {:4f}, recall: {:4f},'\
		' accuracy: {:4f}'.format(overall_res[0], overall_res[1], 
		overall_res[2]))

if __name__ == '__main__':
	simple_per_person(); print('\n')
	simple_inter_person(); print('\n')
	plain_transfer_person(allow_peeking=False, group='similar'); print('\n')
	smart_transfer_person(method='mtpda', group='similar'); print('\n')
	smart_transfer_person(method='tpda', group='similar'); print('\n')
	plain_transfer_person(allow_peeking=True, group='similar'); print('\n')
	plain_transfer_person(allow_peeking=False, group='diff'); print('\n')
	smart_transfer_person(method='mtpda', group='diff'); print('\n')
	smart_transfer_person(method='tpda', group='diff'); print('\n')
	plain_transfer_person(allow_peeking=True, group='diff'); print('\n')
	
	
	
	