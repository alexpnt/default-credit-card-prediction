#!/usr/bin/env python2
# -*- coding: utf-8 -*

from sklearn import preprocessing
from unbalanced_dataset.under_sampling import UnderSampler,NearMiss,NeighbourhoodCleaningRule
from unbalanced_dataset.over_sampling import OverSampler,SMOTE

verbose = True

def standardize(X,axis=0):
	"""
	Scale data to zero mean and unit variance.

	Keyword arguments:
	X -- The feature vectors
	axis -- Default is zero. If axis is 0, standardize each feature, otherwise standardize each input sample
	"""
	return preprocessing.scale(X,axis)

def standardize_features(X):
	"""
	Remove the mean to each feature and scale it to the unit variance.

	Keyword arguments:
	X -- The feature vectors
	"""

	if verbose:
		print '\nStandardizing data ...'

	scaler = preprocessing.StandardScaler().fit(X)
	return scaler.transform(X) 

def scale_by_min_max(X):
	"""
	Scale feature to the given range. Default is 0 - 1.

	Keyword arguments:
	X -- The feature vectors
	feature_range -- The scaling range
	"""

	if verbose:
		print '\nScaling to the range [0-1] ...'

	min_max_scaler = preprocessing.MinMaxScaler()
	return min_max_scaler.fit_transform(X)

def scale_by_max_value(X):
	"""
	Scale each feature by its abs maximum value.

	Keyword arguments:
	X -- The feature vectors	
	"""

	if verbose:
		print '\nScaling to the range [-1,1] ...'

	max_abs_scaler = preprocessing.MaxAbsScaler()
	return max_abs_scaler.fit_transform(X)

def normalize(X,axis=1):
	"""
	Normalize each sample.

	Keyword arguments:
	X -- The feature vectors
	axis -- Default is one. If axis is 0, standardize each feature, otherwise standardize each input sample
	"""
	if verbose:
		if axis==1:
			print '\nNormalizing samples ...'
		else:
			print '\nNormalizing features ...'
	return preprocessing.normalize(X,axis=axis)

def random_undersampling(X,y):
	"""
	Perform random undersampling

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	"""

	if verbose:
		print '\nRandom Majority Undersampling ...'

	undersampler=UnderSampler(verbose=verbose)
	X_undersampled,y_undersampled = undersampler.fit_transform(X,y)
	return X_undersampled,y_undersampled

def nearmiss_undersampling(X,y,version):
	"""
	Perform NearMiss undersampling

	Keyword arguments:
	X -- The feature vectors
	y -- The target classes
	"""

	if verbose:
		print '\nUndersampling with NearMiss-'+str(version)+' ...'

	undersampler=NearMiss(verbose=verbose,version=version)
	X_undersampled,y_undersampled = undersampler.fit_transform(X,y)
	return X_undersampled,y_undersampled

def ncl_undersampling(X,y):
	"""
	Perform the Neighbourhood Cleaning Rule undersampling

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	"""

	if verbose:
		print '\nUndersampling with the Neighbourhood Cleaning Rule ...'

	undersampler=NeighbourhoodCleaningRule(verbose=verbose)
	X_undersampled,y_undersampled = undersampler.fit_transform(X,y)
	return X_undersampled,y_undersampled

def random_oversampling(X,y):
	"""
	Perform random oversampling

	Keyword arguments:
	X -- The feature vectors
	y -- The target classes
	"""

	if verbose:
		print '\nRandom Minority Oversampling ...'
	over_sampler=OverSampler(verbose=verbose)
	X_over_sampled,y_over_sampled = over_sampler.fit_transform(X,y)
	return X_over_sampled,y_over_sampled

def smote_oversampling(X,y):
	"""
	Perform the SMOTE oversampling

	Keyword arguments:
	X -- The feature vectors
	y -- The target classes
	"""

	if verbose:
		print '\nOversampling with SMOTE ...'
	over_sampler=SMOTE(verbose=verbose)
	X_over_sampled,y_over_sampled = over_sampler.fit_transform(X,y)
	return X_over_sampled,y_over_sampled





