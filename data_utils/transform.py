"""
This contains code to take data that was read in by tsio and converts 
it to a standard form which can be used for other things subsequently.
"""
import sys
import pandas as pd
import numpy as np
import settings

def struct_data(data_dict):
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