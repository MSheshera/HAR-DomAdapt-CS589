"""
Code to read in the time series data from disk into pandas dataframes.
This code doesn't work perfectly for every combination of sensor, 
activity and position, this is mostly because of the data itself 
not being perfectly consistent.
This code cant read data for users 4,6,7 and 14; they seem to either have extra
data or lesser data (I think in the case of 6)
"""
import os, sys, pprint, re
import collections
import zipfile as zf
import numpy as np
import pandas as pd

import settings

DEBUG = False

def read_data(root_path, user=['1'], sensor=['acc'], 
	position=['thigh']):
	"""
	Code to read in the data given the path to the root directory.
	Input:
		root_path: path to top-most directory. Sub directories are for 
		the users. Directories below root_path need to be like:
			root_path/proband[num]/data/sen_activity_csv.zip
		user: list of strings saying which users data to read. 
			can be from '1' to '15'.
		sensor: List of strings saying which sensor data to read.
			cab be: ['acc', 'mic', 'mag', 'lig', 'gyr', 'gps']
		position: List of strings saying which position of sensor to read.
			can be: ['chest', 'forearm', 'head', 'shin', 'thigh', 
			'upperarm', 'waist']
	Returns:
		readin_dd: A dict of dicts of dicts of dataframes. Keyed first by 
			user string, next keyed by sensor type, next by activity and 
			then by sensor position. Each value at the end is a pandas 
			dataframe with all the data and final column being an 
			int-mapped label.
	"""
	# Just append 'proband' to each user string.
	user = ['proband'+s for s in user]
	
	# Create directory paths for each file and read it.
	readin_dd = dict()
	for user_str in user:
		temp_sens = dict()
		for sen_str in sensor:
			temp_acts = dict()
			for act_str in settings.act_li:
				# Manually building the path to desired file.
				fname = os.path.join(root_path, user_str,'data',sen_str+'_'+act_str+'_csv.zip')
				if DEBUG is True:
					print user_str, sen_str, act_str
				temp_acts[act_str] = read_zipped_data(fname, position, act_str, sen_str)
			temp_sens[sen_str] = temp_acts
		readin_dd[user_str] = temp_sens

	return readin_dd

def read_zipped_data(fname, position, act_str, sen_str):
	"""
	Given the path to the zipped file this function reads in the data
	for the asked sensor position.
	Input:
		fname: Path to a zipped file.
		position: A list of strings saying which position of sensor to 
			read.
	Returns:
		readin_dict: A dictionary keyed by the sensor position strings
			with values as pandas dataframes of the appropriate data.
	"""
	# Second to last element in the zipped file name is class label.
	int_label = settings.act_map[act_str]
	int_sen = settings.sen_map[sen_str]

	# Initialize ZipFile object.
	data_zf = zf.ZipFile(fname)
	data_zf_files = data_zf.namelist()

	# Return a dictionary with read in data keyed by sensor position.
	readin_dict = dict()
	# Pick csv files which have the sensor positions you want.
	for sen_pos_str in position:
		int_sen_pos = settings.pos_map[sen_pos_str]
		# There will always be a match.
		desired_file = [s for s in data_zf_files if sen_pos_str in s][0]
		with data_zf.open(desired_file, 'rU') as dzf:
			readin = pd.read_csv(dzf, index_col=1)
			# Append sensor string to each column.
			readin.columns = [sen_str+'_'+s for s in readin.columns]
			# Convert time stamps into datetime format.
			readin.index = pd.to_datetime(readin.index, unit='ms')
			# Add a column to indicate sensor, sensor position and label.
			readin['sen'] = pd.Series(np.ones(readin.shape[0])*int_sen).values
			readin['sen_pos'] = pd.Series(np.ones(readin.shape[0])*int_sen_pos).values
			readin['label'] = pd.Series(np.ones(readin.shape[0])*int_label).values
			# Drop first id column.
			readin.drop(readin.columns[0], inplace=True, axis=1)
			if DEBUG is True:
				print sen_pos_str, readin.shape
		readin_dict[sen_pos_str] = readin

	return readin_dict

if __name__ == '__main__':
	if DEBUG is True:
		print('\n\n\n')
		readin = read_data(settings.data_path, user=['1'], sensor=['acc','mic'], position=['thigh'])
		for user in readin.keys():
			for sen in readin[user].keys():
				for act in readin[user][sen].keys():
					for pos in readin[user][sen][act].keys():
						print user, sen, act, pos
						print readin[user][sen][act][pos]