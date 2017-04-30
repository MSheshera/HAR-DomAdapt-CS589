"""
Given time series data chunk and extract features.
"""
import pandas as pd
import numpy as np
from scipy import stats
import settings

def feat_extract(data_df, stat_feat_list=['mean'], 
	comp_feat_list=['pitch','roll','tilt']):
	"""
	Given a data frame with all the data from the different sensors, 
	apply a window and extract the features asked for. Mean is 
	returned all the time.
	"""
	# Always find the mean.
	if 'mean' not in stat_feat_list:
		stat_feat_list.append('mean')
	# Use resample for each activity so that the time gaps 
	# between individual activities don't screw up your 
	# resampling.
	resampled_list = list()
	for int_act in settings.act_map.values():
		temppd = data_df[data_df['label']==int_act]
		# Get the labels data into a labels frame and sample this 
		# data by just returning the first labels in the window.
		label_data = temppd[['label','sen','sen_pos']]
		label_data = label_data.resample('2S',label='right',
			closed='right').apply(ret_first)
		# Get the sensor data into a sensor frame.
		sen_data = temppd[[col for col in temppd.columns if col \
			not in ['label','sen','sen_pos']]]
		# Extract features asked for, for this data.
		feat_pd_list = list()
		if 'max' in stat_feat_list:
			maximum = sen_data.resample('2S',label='right',
				closed='right').max()
			maximum.columns = [s+'_max' for s in maximum.columns]
			feat_pd_list.append(maximum)
		if 'min' in stat_feat_list:
			minimum = sen_data.resample('2S',label='right',
				closed='right').min()
			minimum.columns = [s+'_min' for s in minimum.columns]
			feat_pd_list.append(minimum)
		if 'kurt' in stat_feat_list:
			kurtosis = sen_data.resample('2S',label='right',
				closed='right').apply(kurtosis_f)
			kurtosis.columns = [s+'_kurt' for s in kurtosis.columns]
			feat_pd_list.append(kurtosis)
		if 'skew' in stat_feat_list:
			skew = sen_data.resample('2S',label='right',
				closed='right').apply(skew_f)
			skew.columns = [s+'_skew' for s in skew.columns]
			feat_pd_list.append(skew)
		if 'std' in stat_feat_list:
			stddev = sen_data.resample('2S', label='right',
				closed='right').std()
			stddev.columns = [s+'_std' for s in stddev.columns]
			feat_pd_list.append(stddev)
		# Get the composite features at this point so their mean is
		# is the only thing that gets computed.
		if 'roll' in comp_feat_list:
			sen_data = sen_data.assign(roll = lambda e:np.arctan(e.acc_attr_y/np.sqrt(e.acc_attr_z**2+e.acc_attr_x**2)))
		if 'tilt' in comp_feat_list:
			sen_data = sen_data.assign(tilt = lambda e:np.arccos(e.acc_attr_x/np.sqrt(e.acc_attr_z**2+e.acc_attr_y**2+e.acc_attr_x**2)))
		if 'pitch' in comp_feat_list:
			sen_data = sen_data.assign(pitch = lambda e:np.arctan(e.acc_attr_y/e.acc_attr_z))

		if 'mean' in stat_feat_list:
			mean = sen_data.resample('2S',label='right',
				closed='right').mean()
			mean.columns = [s+'_mean' for s in mean.columns]
			feat_pd_list.append(mean)
		# Add labels into feat_pd_list so it can be concated.
		feat_pd_list.append(label_data)
		# Concat all features extracted.
		temppd = pd.concat(feat_pd_list, axis=1, join='outer')
		# Concat dataframes for all activities.
		resampled_list.append(temppd)
	resampled_cat = pd.concat(resampled_list, axis=0, join='outer')
	resampled_cat.sort_index(inplace=True)
	# Just check that things are happening the way they should
	assert(data_df.shape[0] > resampled_cat.shape[0])
	return resampled_cat

def kurtosis_f(array_like):
	"""
	Kurtosis over the window.
	"""
	return stats.kurtosis(array_like, axis=0, bias=True, 
		nan_policy='omit')

def skew_f(array_like):
	return array_like.skew(axis=0)

def ret_first(array_like):
	"""
	Method for sampling the label information.
	"""
	return array_like.ix[0]	