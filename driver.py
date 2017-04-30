"""
Calls code from everywhere else and gets things done.
"""
import os, sys, pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
from scipy import stats

from data_utils import tsio
from data_utils import settings
from data_utils import transform
from data_utils import feat_extract as fe
from models import dom_adapt

def get_user_asfeats(usr_str):
	"""
	Calls data read and structuring functions to get data of user specified.
	['mean', 'max', 'min', 'kurt', 'std', 'skew']
	"""
	# Read user data in from the zipped directories.
	readin = tsio.read_data(settings.data_path, user=[usr_str], 
		sensor=['acc','gyr'], position=['thigh'])

	# Get data for all sensors and poisitions into one 
	# consistently structured dataframe. Doing single user for now.
	for user in readin.keys():
		user_data = transform.struct_data(readin[user])

	# Extract features from the data read in.
	feat_user_data = fe.feat_extract(user_data, 
		stat_feat_list=['mean', 'max', 'min', 'kurt', 'std', 'skew'])
	return feat_user_data

# Learn on one person and test on the same person.
def simple_per_person():
	# Read and structure data appropriately.
	user_data = get_user_asfeats(usr_str='2')
	# Create test, train split.
	train, test = model_selection.train_test_split(user_data, 
		test_size=0.2)
	train_y = train['label']
	test_y = test['label']
	train_x = train.drop('label', axis=1)
	test_x = test.drop('label', axis=1)
	print 'Train shape:', train_x.shape
	print 'Test shape:', test_x.shape

	# Train model.
	rf_clf = ensemble.RandomForestClassifier(n_estimators=10, 
		criterion='entropy', max_features='sqrt')
	rf_clf.fit(train_x, train_y)
	print 'Fit model:', rf_clf.feature_importances_

	# Test on test data.
	preds = rf_clf.predict(test_x)
	print 'Predicted.'

	# Evaluate the results.
	print metrics.classification_report(test_y, preds, 
		labels=range(1,len(settings.act_li)+1,1), target_names=settings.act_li)

# Learn on multiple people and test on same set of people.
def simple_inter_person():
	user_data_frames = list()
	for usr_str in ['1','2','3']:
		resampled_data = get_user_asfeats(usr_str)
		user_data_frames.append(resampled_data)
	user_data = pd.concat(user_data_frames, axis=0, join='outer')
	print 'Train data:', user_data.shape
	sys.stdout.flush()
	
	train, test = model_selection.train_test_split(user_data, 
		test_size=0.2)

	# Create test, train split.
	train_y = train['label']
	test_y = test['label']
	train_x = train.drop('label', axis=1)
	test_x = test.drop('label', axis=1)
	print 'Train shape:', train_x.shape
	print 'Test shape:', test_x.shape
	sys.stdout.flush()

	# Train model.
	rf_clf = ensemble.RandomForestClassifier(n_estimators=10, 
		criterion='entropy', max_features='sqrt')
	rf_clf.fit(train_x, train_y)
	print 'Fit model:', rf_clf.feature_importances_
	sys.stdout.flush()

	# Test on test data.
	preds = rf_clf.predict(test_x)
	print 'Predicted.'

	# Evaluate the results.
	print metrics.classification_report(test_y, preds, 
		labels=range(1,len(settings.act_li)+1,1), target_names=settings.act_li)

# Learn on some people and test on entirely different person.
def plain_transfer_person():
	user_data_frames = list()
	for usr_str in ['1','2','3']:
		resampled_data = get_user_asfeats(usr_str)
		user_data_frames.append(resampled_data)
	train_users = pd.concat(user_data_frames, axis=0, join='outer')
	print 'Train data:', train_users.shape
	sys.stdout.flush()
	
	user_data_frames = list()
	for usr_str in ['8']:
		resampled_data = get_user_asfeats(usr_str)
		user_data_frames.append(resampled_data)
	test_users = pd.concat(user_data_frames, axis=0, join='outer')
	print 'Test data:', test_users.shape
	sys.stdout.flush()

	# Drop sen and sen_pos labels
	train_users = train_users.drop('sen', axis=1)
	train_users = train_users.drop('sen_pos', axis=1)
	test_users = test_users.drop('sen', axis=1)
	test_users = test_users.drop('sen_pos', axis=1)

	# Let the training see some of the target data.
	test_subset, test_users = model_selection.train_test_split(
		test_users, test_size=0.9)
	train_users = pd.concat([train_users, test_subset], axis=0)
	print 'Train data:', train_users.shape
	sys.stdout.flush()

	# Create test, train split.
	train_y = train_users['label']
	test_y = test_users['label']
	train_x = train_users.drop('label', axis=1)
	test_x = test_users.drop('label', axis=1)
	print 'Train shape:', train_x.shape
	print 'Test shape:', test_x.shape

	# Train model.
	rf_clf = ensemble.RandomForestClassifier(n_estimators=10, 
		criterion='entropy', max_features='sqrt', random_state=34)
	rf_clf.fit(train_x, train_y)
	print 'Fit model:', rf_clf.feature_importances_

	# Test on test data.
	preds = rf_clf.predict(test_x)
	print 'Predicted.'

	# Evaluate the results.
	print metrics.classification_report(test_y, preds, 
		labels=range(1,len(settings.act_li)+1,1), target_names=settings.act_li)

# Attempting some transfer learning.
def smart_transfer_person(read_disk, write_disk):
	# Read in source domain data either extracted or perform
	# extraction.
	if read_disk is True:
		with open('./data/source-123.pd','rb') as f:
			source_data = pickle.load(f)
		with open('./data/target-8.pd','rb') as f:
			tar_data = pickle.load(f)
	else:
		user_data_frames = list()
		for usr_str in ['1','2','3']:
			resampled_data = get_user_asfeats(usr_str)
			user_data_frames.append(resampled_data)
		source_data = pd.concat(user_data_frames, axis=0, join='outer')
		# Drop sensor position and sensor type labels. (quite useless)
		source_data = source_data.drop('sen', axis=1)
		source_data = source_data.drop('sen_pos', axis=1)

		# Read in target domain data.
		user_data_frames = list()
		for usr_str in ['8']:
			resampled_data = get_user_asfeats(usr_str)
			user_data_frames.append(resampled_data)
		tar_data = pd.concat(user_data_frames, axis=0, join='outer')
		# Drop sensor position and sensor type labels. 
		tar_data = tar_data.drop('sen', axis=1)
		tar_data = tar_data.drop('sen_pos', axis=1)

		if write_disk is True:
			source_data.to_pickle('./data/source-123.pd')
			tar_data.to_pickle('./data/target-8.pd')

	print 'Source data:', source_data.shape
	sys.stdout.flush()
	
	print 'Target data:', tar_data.shape
	sys.stdout.flush()

	# Form small portion of target data for training.
	tar_data_tr, tar_data_te = model_selection.train_test_split(
		tar_data, test_size=0.8)

	
	# Do the tpda thing.
	adapted_clfs, mappers = dom_adapt.modified_tpda(tar_data_tr, source_data)
	target_y = tar_data['label']
	target_x = tar_data.drop('label', axis=1)
	predictions = list()
	for clf, mapper in zip(adapted_clfs, mappers):
		target_x_ld = mapper.transform(target_x)
		# Test on target data.
		preds = clf.predict(target_x_ld)
		predictions.append(preds.reshape(preds.shape[0],1))
	predictions = np.concatenate(predictions, axis=1)
	predictions = stats.mode(predictions, axis=1).mode


	# Evaluate the results.
	print metrics.classification_report(target_y, predictions, 
		labels=range(1,len(settings.act_li)+1,1), target_names=settings.act_li)

if __name__ == '__main__':
	read_disk = True
	write_disk = False
	smart_transfer_person(read_disk, write_disk)

	#plain_transfer_person()