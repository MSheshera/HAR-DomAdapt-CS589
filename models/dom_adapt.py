"""
Implementing the Topilogy Preserving Domain Adoptation.
https://pdfs.semanticscholar.org/881d/78226a8e05bb4cad13d4e833f5a73a61eb77.pdf
"""
import numpy as np
import pandas as pd

from sklearn import manifold
from sklearn import neighbors
from sklearn import ensemble

from data_utils import settings
DEBUG = False

class TopologyPreservingDA:
	pass

def tpda(tar_data_tr, source_data):
	"""
	Fits a tpda model.
	"""
	# Get labels and data seperated.
	tar_data_labels = tar_data_tr['label']
	tar_data_tr = tar_data_tr.drop('label', axis=1)
	source_data_labels = source_data['label']
	source_data = source_data.drop('label', axis=1)

	# Using embedding size of 15 arbitrarily now.
	isomapper = manifold.Isomap(n_neighbors=5, n_components=15, 
		eigen_solver='auto', tol=0, max_iter=None, 
		path_method='auto', neighbors_algorithm='auto', n_jobs=2)
	# Compute embeddings for small target set.
	isomapper.fit(tar_data_tr)
	tar_data_tr_ld = isomapper.transform(tar_data_tr)
	if DEBUG is True:
		print('tpda: Computed low dim projection of target set. {} {}'.format(tar_data_tr_ld.shape, type(tar_data_tr_ld)))
	
	# Transform source data to lower dimension space learnt on target
	# domain.
	source_data_ld = isomapper.transform(source_data)
	if DEBUG is True:
		print('tpda: Mapped source domain to target set projection.'.format(source_data_ld.shape, type(source_data_ld)))

	# Build selected dataset.
	## Get neighbourhoods for each class in the source dataset.
	act_neigbourhoods = dict()
	for int_act in settings.act_map.values():
		act_subset = source_data_ld[source_data_labels==int_act]
		neigh = neighbors.NearestNeighbors(n_neighbors=20, 
			metric='minkowski', p=2, metric_params=None, n_jobs=2)
		neigh.fit(act_subset)
		act_neigbourhoods[int_act] = neigh

	## For each class in labelled target subset get nearest neighbours
	## from source dataset.
	selected_act_list = list()
	tar_act_list = list()
	for int_act in settings.act_map.values():
		act_subset = source_data_ld[source_data_labels==int_act]
		act_subset_tar = tar_data_tr_ld[tar_data_labels==int_act]
		# Get the appropriate data structure.
		act_neigh = act_neigbourhoods[int_act]
		# Get indices for nn n_neighbors neighbours of all samples in 
		# act_subset_tar.
		neigh_indices = act_neigh.kneighbors(act_subset_tar, 
			return_distance=False)
		# Flatten indices.
		neigh_indices = neigh_indices.flatten(order='C')
		# Index into source data.
		selected = act_subset[neigh_indices, :]
		# Append until-now-removed label to selected subset.
		selected = pd.DataFrame(selected)
		selected['label'] = pd.Series(np.ones(selected.shape[0])*int_act).values
		selected_act_list.append(selected)
		# Build a list of dataframes of the target subset also.
		act_tar = pd.DataFrame(act_subset_tar)
		act_tar['label'] = pd.Series(np.ones(act_tar.shape[0])*int_act).values
		tar_act_list.append(act_tar)
		if DEBUG is True:
			print('act: {}'.format(int_act))
			print('act_subset_tar: {}'.format(act_subset_tar.shape))
			print('neigh_indices: {}'.format(neigh_indices.shape))
			print('neigh_indices_flat: {}'.format(neigh_indices.shape))
			print('selected: {}'.format(selected.shape))
			print('\n')
		
	source_selected = pd.concat(selected_act_list, axis=0, 
		ignore_index=True)
	tar_data_tr_ld = pd.concat(tar_act_list, axis=0, 
		ignore_index=True)
	print('Source selected: {}'.format(source_selected.shape))
	print('Target low dim: {}'.format(tar_data_tr_ld.shape))
	# Concat selected and target subset.
	training = pd.concat([source_selected, tar_data_tr_ld], axis=0, 
		ignore_index=True)
	# Train a random forest classifier on selected source data.
	rf_clf = ensemble.RandomForestClassifier(n_estimators=10, 
		criterion='entropy', max_features='sqrt')
	train_y = training['label']
	train_x = training.drop('label', axis=1)
	rf_clf.fit(train_x, train_y)
	return rf_clf, isomapper