"""
Impliments Topology Preserving Domain Adoptation as it is described here:
https://pdfs.semanticscholar.org/881d/78226a8e05bb4cad13d4e833f5a73a61eb77.pdf
Also impliments a variant of the same method.
"""
import os, sys

import numpy as np
import pandas as pd
from scipy import stats

from sklearn import manifold
from sklearn import neighbors
from sklearn import ensemble

from data_utils import settings
DEBUG = False

class TopologyPreservingDA:
	"""
	This impliments TPDA exactly as it is described in the paper.
	"""
	def __init__(self, n_iter=5, isomap_comp=15, nn_neighbors=10, 
		random_state=46, clf_str='rf', verbose=False):
		"""
		Initialize n_iter number of mappers and classifiers.
		"""
		self.n_iter = n_iter
		self.verbose = verbose
		self.nn_neighbors = nn_neighbors
		self.mappers = list()
		self.classifiers = list()
		for i in range(n_iter):
			self.mappers.append(manifold.Isomap(n_neighbors=5, 
				n_components=isomap_comp, eigen_solver='auto', 
				tol=0, max_iter=None, path_method='auto', 
				neighbors_algorithm='auto', n_jobs=2))
			# Potentially allow for other classifiers to be used.
			# But supports only random forests for now.
			if clf_str is 'rf':
				self.classifiers.append(ensemble.RandomForestClassifier(
					n_estimators=10, criterion='entropy', max_features='sqrt', 
					random_state = random_state))

	def fit(self, tar_data_subset, source_data):
		"""
		Fits the model. Takes a subset of the labelled target domain data and
		and the labelled source data. Both of these are assumed to be pandas
		dataframes.
		"""
		# Get labels and data seperated.
		tar_data_labels = tar_data_subset['label']
		tar_data_tr = tar_data_subset.drop('label', axis=1)
		source_data_labels = source_data['label']
		source_data = source_data.drop('label', axis=1)

		mapper_data = tar_data_tr
		for iteration in range(self.n_iter):
			if self.verbose:
				print('Iteration: {}'.format(iteration))
				print('Mapper data: {}'.format(mapper_data.shape))
			sys.stdout.flush()
			# Get the mapper.
			isomapper = self.mappers[iteration]
			# Compute embeddings for small target set.
			isomapper.fit(mapper_data)
			if iteration == 0:
				tar_data_tr_ld = isomapper.transform(tar_data_tr)
				if self.verbose:
					print('Computed low dim projection of target set. {}'
						.format(tar_data_tr_ld.shape))
			sys.stdout.flush()
			
			# Transform source data to lower dimension space learnt on target
			# domain.
			source_data_ld = isomapper.transform(source_data)
			if self.verbose:
				print('Mapped source domain to target set projection. {}'
					.format(source_data_ld.shape))
			sys.stdout.flush()

			# Build selected dataset.
			## Get neighbourhoods for each class in the source dataset.
			act_neigbourhoods = dict()
			for int_act in settings.act_map.values():
				act_subset_ld = source_data_ld[source_data_labels==int_act]
				neigh = neighbors.NearestNeighbors(
					n_neighbors=self.nn_neighbors, metric='minkowski', p=2, 
					metric_params=None, n_jobs=2)
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
				selected_ld['label'] = pd.Series(np.ones(
					selected_ld.shape[0])*int_act).values
				selected_ld_act_list.append(selected_ld)
				selected_fd_act_list.append(selected_fd)
				if self.verbose:
					print('act: {}'.format(int_act))
					print('act_subset_tar: {}'.format(act_subset_tar.shape))
					print('neigh_indices_flat: {}'.format(neigh_indices.shape))
					print('selected: {}'.format(selected_ld.shape))
				
			source_selected_ld = pd.concat(selected_ld_act_list, axis=0, 
				ignore_index=True)
			tar_data_ld = pd.DataFrame(tar_data_tr_ld)
			tar_data_ld['label'] = tar_data_labels.values
			if self.verbose:
				print('Source selected_ld: {}'.format(source_selected_ld.shape))
				print('Target low dim: {}'.format(tar_data_ld.shape))
			# Concat selected and target subset.
			training = pd.concat([source_selected_ld, tar_data_ld], axis=0, 
				ignore_index=True)
			# Train a classifier on low dim selected source data.
			rf_clf = self.classifiers[iteration]
			train_y = training['label']
			train_x = training.drop('label', axis=1)
			if self.verbose:
				print('Train_x: {}'.format(train_x.shape))
			rf_clf.fit(train_x, train_y)

			# Get selected full dimensional data for learning the mapper 
			# in the next iteration.
			source_selected_fd = pd.concat(selected_fd_act_list, axis=0, 
				ignore_index=True)
			if self.verbose:
				print("source_selected_fd: {}".format(source_selected_fd.shape))
			mapper_data = pd.concat([source_selected_fd, tar_data_tr], axis=0, 
				ignore_index=True)
			if self.verbose:
				print('\n')

	def predict(self, target_x):
		"""
		Make predictions with the learnt mapper and classifier.
		"""
		predictions = list()
		adapted_clfs = self.classifiers
		mappers = self.mappers
		for clf, mapper in zip(adapted_clfs, mappers):
			target_x_ld = mapper.transform(target_x)
			# Test on target data.
			preds = clf.predict(target_x_ld)
			predictions.append(preds.reshape(preds.shape[0],1))
		predictions = np.concatenate(predictions, axis=1)
		predictions = stats.mode(predictions, axis=1).mode
		return predictions


class modTopologyPreservingDA:
	"""
	This impliments TPDA without looking at the labels in the target subset.
	"""
	def __init__(self, n_iter=1, isomap_comp=15, nn_neighbors=10, 
		random_state=46, clf_str='rf', verbose=False):
		"""
		Initialize n_iter number of mappers and classifiers. 
		Avoids the n_iterations part from the paper. Functionality for 
		n_iter only for experimentation.
		"""
		self.n_iter = n_iter
		self.verbose = verbose
		self.nn_neighbors = nn_neighbors
		self.mappers = list()
		self.classifiers = list()
		for i in range(n_iter):
			self.mappers.append(manifold.Isomap(n_neighbors=5, 
				n_components=isomap_comp, eigen_solver='auto', 
				tol=0, max_iter=None, path_method='auto', 
				neighbors_algorithm='auto', n_jobs=2))
			# Potentially allow for other classifiers to be used.
			# But supports only random forests for now.
			if clf_str is 'rf':
				self.classifiers.append(ensemble.RandomForestClassifier(
					n_estimators=10, criterion='entropy', max_features='sqrt', 
					random_state = random_state))

	def fit(self, tar_data_tr, source_data):
		"""
		Fits the model. Takes a subset of the unlabelled target domain data and
		and the labelled source data. Both of these are assumed to be pandas
		dataframes.
		"""
		# Get labels and data seperated for source data; target subset 
		# lacks labels.
		source_data_labels = source_data['label']
		source_data = source_data.drop('label', axis=1)

		mapper_data = tar_data_tr
		for iteration in range(self.n_iter):
			if self.verbose:
				print('Iteration: {}'.format(iteration))
				print('Mapper data: {}'.format(mapper_data.shape))
			sys.stdout.flush()
			# Get the mapper.
			isomapper = self.mappers[iteration]
			# Compute embeddings for small target set.
			isomapper.fit(mapper_data)
			if iteration == 0:
				tar_data_tr_ld = isomapper.transform(tar_data_tr)
				if self.verbose:
					print('Computed low dim projection of target set. {}'
						.format(tar_data_tr_ld.shape))
			sys.stdout.flush()
			
			# Transform source data to lower dimension space learnt on target
			# domain.
			source_data_ld = isomapper.transform(source_data)
			if self.verbose:
				print('Mapped source domain to target set projection. {}'
					.format(source_data_ld.shape))
			sys.stdout.flush()

			# Build selected dataset.
			## Get neighbourhood for the source dataset.
			neigh = neighbors.NearestNeighbors(
				n_neighbors=self.nn_neighbors, metric='minkowski', p=2, 
				metric_params=None, n_jobs=2)
			neigh.fit(source_data_ld)

			## For each sample in ld target subset get nearest neighbours
			## from ld source dataset.
			neigh_indices = neigh.kneighbors(tar_data_tr_ld, return_distance=False)
			## Flatten indices.
			neigh_indices = neigh_indices.flatten(order='C')
			## Index into fd source data for building mapper data
			selected_fd = source_data.ix[neigh_indices]
			## Index into ld source data.
			selected_ld = source_data_ld[neigh_indices, :]
			## Append until-now-removed label to selected subset.
			selected_ld = pd.DataFrame(selected_ld)
			selected_ld['label'] = source_data_labels.values[neigh_indices]
			if self.verbose is True:
				print('neigh_indices_flat: {}'.format(neigh_indices.shape))
				print('selected_ld: {}'.format(selected_ld.shape))
				
			# Train a random forest classifier on selected source data.
			rf_clf = self.classifiers[iteration]
			train_y = selected_ld['label']
			train_x = selected_ld.drop('label', axis=1)
			rf_clf.fit(train_x, train_y)

			mapper_data = pd.concat([selected_fd, tar_data_tr], axis=0, 
				ignore_index=True)
			if self.verbose:
				print('\n')

	def predict(self, target_x):
		"""
		Make predictions with the learnt mapper and classifier.
		"""
		predictions = list()
		adapted_clfs = self.classifiers
		mappers = self.mappers
		for clf, mapper in zip(adapted_clfs, mappers):
			target_x_ld = mapper.transform(target_x)
			# Test on target data.
			preds = clf.predict(target_x_ld)
			predictions.append(preds.reshape(preds.shape[0],1))
		predictions = np.concatenate(predictions, axis=1)
		predictions = stats.mode(predictions, axis=1).mode
		return predictions