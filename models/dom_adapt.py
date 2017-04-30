"""
Implementing the Topilogy Preserving Domain Adoptation.
https://pdfs.semanticscholar.org/881d/78226a8e05bb4cad13d4e833f5a73a61eb77.pdf
"""
import os, sys
import numpy as np
import pandas as pd

from sklearn import manifold
from sklearn import neighbors
from sklearn import ensemble

from data_utils import settings
DEBUG = True

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

	mapper_data = tar_data_tr
	clfs = list()
	mappers = list()
	for i in range(5):
		print('Iteration: {}'.format(i))
		print('Mapper data: {}'.format(mapper_data.shape))
		sys.stdout.flush()
		# Using embedding size of 15 arbitrarily now.
		isomapper = manifold.Isomap(n_neighbors=5, n_components=15, 
			eigen_solver='auto', tol=0, max_iter=None, 
			path_method='auto', neighbors_algorithm='auto', n_jobs=2)
		# Compute embeddings for small target set.
		isomapper.fit(mapper_data)
		tar_data_tr_ld = isomapper.transform(tar_data_tr)
		if DEBUG is True:
			print('tpda: Computed low dim projection of target set. {} {}'.format(tar_data_tr_ld.shape, type(tar_data_tr_ld)))
			sys.stdout.flush()
		
		# Transform source data to lower dimension space learnt on target
		# domain.
		source_data_ld = isomapper.transform(source_data)
		if DEBUG is True:
			print('tpda: Mapped source domain to target set projection.'.format(source_data_ld.shape, type(source_data_ld)))
			sys.stdout.flush()

		# Build selected dataset.
		## Get neighbourhoods for each class in the source dataset.
		act_neigbourhoods = dict()
		for int_act in settings.act_map.values():
			act_subset_ld = source_data_ld[source_data_labels==int_act]
			neigh = neighbors.NearestNeighbors(n_neighbors=10, 
				metric='minkowski', p=2, metric_params=None, n_jobs=2)
			neigh.fit(act_subset_ld)
			act_neigbourhoods[int_act] = neigh

		## For each class in labelled target subset get nearest neighbours
		## from source dataset.
		selected_ld_act_list = list()
		selected_fd_act_list = list()
		for int_act in settings.act_map.values():
			act_subset_ld = source_data_ld[source_data_labels==int_act]
			act_subset_fd = source_data[source_data_labels==int_act]
			act_subset_tar = tar_data_tr_ld[tar_data_labels==int_act]
			# Get the appropriate data structure.
			act_neigh = act_neigbourhoods[int_act]
			# Get indices for nn n_neighbors neighbours of all samples in 
			# act_subset_tar.
			neigh_indices = act_neigh.kneighbors(act_subset_tar, 
				return_distance=False)
			# Flatten indices.
			neigh_indices = neigh_indices.flatten(order='C')
			# Index into low dim source data.
			selected_ld = act_subset_ld[neigh_indices, :]
			# Index into full dim source data
			selected_fd = act_subset_fd.ix[neigh_indices]
			# Append until-now-removed label to selected subset.
			selected_ld = pd.DataFrame(selected_ld)
			selected_ld['label'] = pd.Series(np.ones(selected_ld.shape[0])*int_act).values
			selected_ld_act_list.append(selected_ld)
			selected_fd_act_list.append(selected_fd)
			if False:
				print('act: {}'.format(int_act))
				print('act_subset_tar: {}'.format(act_subset_tar.shape))
				print('neigh_indices_flat: {}'.format(neigh_indices.shape))
				print('selected: {}'.format(selected_ld.shape))
				print('\n')
			
		source_selected_ld = pd.concat(selected_ld_act_list, axis=0, 
			ignore_index=True)
		tar_data_tr_ld = pd.DataFrame(tar_data_tr_ld)
		tar_data_tr_ld['label'] = tar_data_labels.values
		print('Source selected_ld: {}'.format(source_selected_ld.shape))
		print('Target low dim: {}'.format(tar_data_tr_ld.shape))
		# Concat selected and target subset.
		training = pd.concat([source_selected_ld, tar_data_tr_ld], axis=0, 
			ignore_index=True)
		# Train a random forest classifier on low dim selected source data.
		rf_clf = ensemble.RandomForestClassifier(n_estimators=10, 
			criterion='entropy', max_features='sqrt')
		train_y = training['label']
		train_x = training.drop('label', axis=1)
		print('Train_x: {}'.format(train_x.shape))
		rf_clf.fit(train_x, train_y)

		# Get selected full dimensional data for learning mapper in next iteration.
		source_selected_fd = pd.concat(selected_fd_act_list, axis=0, 
			ignore_index=True)
		print("source_selected_fd: {}".format(source_selected_fd.shape))
		mapper_data = pd.concat([source_selected_fd, tar_data_tr], axis=0, 
			ignore_index=True)
		print("mapper_data: {}".format(mapper_data.shape))
		# Append classifier and mapper to list for each iteration.
		clfs.append(rf_clf)
		mappers.append(isomapper)
		print '\n'

	return clfs, mappers

def modified_tpda(tar_data_tr, source_data):
	"""
	Fits a modified tpda model.
	"""
	# Get labels and data seperated.
	tar_data_labels = tar_data_tr['label']
	tar_data_tr = tar_data_tr.drop('label', axis=1)
	source_data_labels = source_data['label']
	# Remove datetime indices and set them to integers so you can
	# index into this later.
	#source_data_labels.index = range(source_data_labels.shape[0])
	source_data = source_data.drop('label', axis=1)

	# Using embedding size of 15 arbitrarily now.
	mapper_data = tar_data_tr
	clfs = list()
	mappers = list()
	for i in range(5):
		isomapper = manifold.Isomap(n_neighbors=5, n_components=5, 
			eigen_solver='auto', tol=0, max_iter=None, 
			path_method='auto', neighbors_algorithm='auto', n_jobs=2)
		# Compute embeddings for small target set.
		isomapper.fit(mapper_data)
		mappers.append(isomapper)
		tar_data_tr_ld = isomapper.transform(tar_data_tr)
		if DEBUG is True:
			print('tpda: Computed low dim projection of target set. {} {}'.format(tar_data_tr_ld.shape, type(tar_data_tr_ld)))
		
		# Transform source data to lower dimension space learnt on target
		# domain.
		source_data_ld = isomapper.transform(source_data)
		if DEBUG is True:
			print('tpda: Mapped source domain to target set projection.'.format(source_data_ld.shape, type(source_data_ld)))

		# Build selected dataset.
		## Build data strcture to find neighbours in source data.
		neigh = neighbors.NearestNeighbors(n_neighbors=10, 
			metric='minkowski', p=2, metric_params=None, n_jobs=2)
		neigh.fit(source_data_ld)

		## For each sample in the target subset find k closest neighbours
		## in the mapped source domain.

		## Get indices for nn n_neighbors neighbours of all samples in 
		## act_subset_tar.
		neigh_indices = neigh.kneighbors(tar_data_tr_ld, return_distance=False)
		## Flatten indices.
		neigh_indices = neigh_indices.flatten(order='C')
		## Index into source data.
		selected_ld = source_data_ld[neigh_indices, :]
		selected_fd = source_data.ix[neigh_indices]
		## Form mapper data for next iteration
		mapper_data = pd.concat([selected_fd, tar_data_tr], axis=0, 
			ignore_index=True)
		print("mapper_data: {}".format(mapper_data.shape))
		## Append until-now-removed label to selected subset.
		selected_ld = pd.DataFrame(selected_ld)
		selected_ld['label'] = source_data_labels.values[neigh_indices]
		if DEBUG is True:
			print('neigh_indices_flat: {}'.format(neigh_indices.shape))
			print('selected: {}'.format(selected_ld.shape))
			print('\n')
			
		# Train a random forest classifier on selected source data.
		rf_clf = ensemble.RandomForestClassifier(n_estimators=10, 
			criterion='entropy', max_features='sqrt', random_state=34)
		train_y = selected_ld['label']
		train_x = selected_ld.drop('label', axis=1)
		rf_clf.fit(train_x, train_y)
		clfs.append(rf_clf)

	return clfs, mappers