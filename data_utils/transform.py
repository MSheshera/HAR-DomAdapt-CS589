"""
This contains code to take data that was read in by tsio and converts 
it to a standard form which can be used for other things subsequently.
"""
import os, sys, pickle
import pandas as pd
import numpy as np

import settings
import tsio
import feat_extract as fe

def read_feats(source_li=['1','2','3'], target_li=['8']):
	"""
	Read extracted features from disk and return them as pandas dataframes.
	If the target_li is None it means that the data is for a non domain
	adaptation kind of an experiment. Drops sensor and sensor position labels
	at all times.
	"""
	# Expects these to be present and named this way.
	data_dir='./data/'
	basename = 'user_'
	# Read source users data
	## Create a list of user dataframes so you can concat them all together.
	src_li = list()
	for usr_str in source_li:
		fname = os.path.join(data_dir, basename+usr_str+'.pd')
		with open(fname,'rb') as f:
			data = pickle.load(f)
		src_li.append(data)
	source_data = pd.concat(src_li, axis=0, join='outer')
	## Drop sensor position and sensor type labels.
	source_data = source_data.drop('sen', axis=1)
	source_data = source_data.drop('sen_pos', axis=1)

	# Read target users data
	if target_li != None:
		tar_li = list()
		for usr_str in target_li:
			fname = os.path.join(data_dir, basename+usr_str+'.pd')
			with open(fname,'rb') as f:
				data = pickle.load(f)
			tar_li.append(data)
		target_data = pd.concat(tar_li, axis=0, join='outer')
		## Drop sensor position and sensor type labels.
		target_data = target_data.drop('sen', axis=1)
		target_data = target_data.drop('sen_pos', axis=1)
	else:
		# If not asked for target just return empty dataframe.
		target_data = pd.DataFrame([])
	
	# At this point just replace nan by zero; might not be the best thing but
	# theres very little I can do. Dont want to hand hold data from each user.
	source_data.fillna(value=0, axis=None, inplace=True)
	target_data.fillna(value=0, axis=None, inplace=True)

	return source_data, target_data

def user_to_disk(data_dir='./data/'):
	"""
	Extract all possible features for all possible users and write it to disk.
	"""
	basename = 'user_'
	for usr_str in settings.usr_li:
		feat_user_data = get_user_asfeats(usr_str)
		fname = os.path.join(data_dir, basename+usr_str+'.pd')
		feat_user_data.to_pickle(fname)
		print('User {}: {}'.format(usr_str, feat_user_data.shape))
		print('Wrote: {}'.format(fname))

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
		user_data = struct_data(readin[user])

	# Extract features from the data read in.
	feat_user_data = fe.feat_extract(user_data, 
		stat_feat_list=['mean', 'max', 'min', 'kurt', 'std', 'skew'],
		comp_feat_list=['pitch','roll','tilt'])
	return feat_user_data

def struct_data(data_dict, resample=True, method='mean'):
	"""
	Given the data dictionary return a gigantic dataframe with all the data
	put together and sorted by timestamp.
	Input: 
		data_dict: A per person dictionary of read in data from tsio. 
	Return:
		data_dict: 
	"""
	# Build a list of the dataframes so you can concat them together.	
	dataframe_list = list()
	for sen in data_dict.keys():
		for act in data_dict[sen].keys():
			for pos in data_dict[sen][act].keys():
				#print user, sen, act, pos
				dataframe_list.append(data_dict[sen][act][pos])

	# Concat the dataframes. 
	catdf = pd.concat(dataframe_list, join='outer')
	# Sort by indices (timestamps)
	catdf.sort_index(inplace=True)
	return catdf

if __name__ == '__main__':
	# Extract all features and write them to disk. Expectes directory called
	# 'data' to be present in current directory; expects permissions to write 
	# to it.
	user_to_disk(data_dir='./data/')