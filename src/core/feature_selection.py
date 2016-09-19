#!/usr/bin/env python2
# -*- coding: utf-8 -*
# 
from __future__ import  division
import sys
sys.path.append("..")
from scipy import stats
from sklearn.feature_selection import chi2
import numpy as np
import math
from utils import split_tuples
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from peng_mrmr.mRMR import mrmr
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

verbose=True

def kolmogorov_smirnov_normality_test(X,y):
	"""
	Performs the one sample Kolmogorov-Smirnov test, testing wheter the feature values of each class are drawn from a normal distribution

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	"""

	kolmogorov_smirnov={}
	# print kolmogorov_smirnov
	for feature_col in xrange(len(X[0])):
		kolmogorov_smirnov[feature_col]=values=[]
		for class_index in xrange(2):
			values.append(stats.kstest(X[y==class_index,feature_col], 'norm'))

		
	#debug
	for f in xrange(23):
			print kolmogorov_smirnov[f]

	return kolmogorov_smirnov

def kolmogorov_smirnov_two_sample_test(X,y):
	"""
	Performs the two sample Kolmogorov-Smirnov test, testing wheter feature values of each class are drawn from identical distributions

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	"""

	kolmogorov_smirnov=[[(0,0)]]*len(X[0])
	# print kolmogorov_smirnov
	for feature_col in xrange(len(X[0])):
			ks_test_statistic,p_value=stats.ks_2samp(X[y==0,feature_col],X[y==1,feature_col])
			kolmogorov_smirnov[feature_col]=(ks_test_statistic,p_value)

	#debug
	for f in xrange(23):
		print kolmogorov_smirnov[f]

	return kolmogorov_smirnov

def kolmogorov_smirnov_two_sample_test(sample_a,sample_b):
	"""
	Performs the two sample Kolmogorov-Smirnov test, testing wheter twoa samples are drawn from identical distributions

	Keyword arguments:
	sample_a -- The first sample
	sample_b -- The second sample
	"""

	return stats.ks_2samp(sample_a,sample_b)

def chi2_feature_test(X,y,feature_index):
	"""
	Performs the chi square test on the desired feature

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	feature_index - The selected feature (a zero-based index)
	"""

	feature_column=X[:,feature_index].reshape(-1,1)
	min_val=feature_column.min()
	if min_val<0:
		feature_column=feature_column+min_val*-1+1
	return chi2(feature_column,y)

def kw_feature_test(X,y,feature_index):
	"""
	Performs the Kruskal-Wallis H-test for desired feature

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	feature_index - The selected feature (a zero-based index)
	"""

	feature_column=X[:,feature_index]
	return stats.mstats.kruskalwallis(feature_column[y==0],feature_column[y==1])

def entropy(data):
	"""
	Computes the Entropy of a set of data

	Keyword arguments:
	data -- vector data
	"""

	unique_values, unique_counts= np.unique(data, return_counts=True)	#counts of each unique value
	probabilities=unique_counts/len(data)
	return -probabilities.dot(np.log(probabilities))

def information_gain(X,y,feature_index):
	"""
	Computes the Information Gain achieved by a feature in relation to the target class

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	feature_index - The selected feature (a integer index)
	"""

	feature_column=X[:,feature_index]
	unique_values, unique_counts= np.unique(feature_column, return_counts=True)	#counts of each unique value
	feature_entropy=0.0
	for i in xrange(len(unique_values)):
		feature_entropy+=unique_counts[i] * entropy(y[feature_column==unique_values[i]])

	return entropy(y) - feature_entropy/len(y)

def gain_ratio(X,y,feature_index):
	"""
	Computes the Gain Ratio achieved by a feature in relation to the target class

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	feature_index - The selected feature (a integer index)
	"""

	feature_column=X[:,feature_index]
	unique_values, unique_counts= np.unique(feature_column, return_counts=True)	#counts of each unique value
	probabilities=unique_counts/len(y)
	split_information=0.0
	for i in xrange(len(unique_values)):
		split_information      += (-probabilities[i]) * math.log(probabilities[i], 2)

	return information_gain(X,y,feature_index)/split_information

def pearson_correlation_matrix(X):
	"""
	Computes the Pearson Correlation matrix

	Keyword arguments:
	X -- The feature vectors
	"""

	n_features=len(X[0])
	correlation_matrix=np.zeros(shape=(n_features,n_features))
	for i in xrange(n_features):
		for j in xrange(n_features):
			pearson_corr=stats.pearsonr(X[:,i],X[:,j])[0]
			correlation_matrix[i][j]=pearson_corr

	return correlation_matrix

def fisher_score(X,y,feature_index):
	"""
	Computes the Fisher Score achieved by a feature in relation to the target class

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	feature_index - The selected feature (a integer index)
	"""

	class1=X[y==0,feature_index]
	class2=X[y==1,feature_index]
	m1=np.mean(class1)
	m2=np.mean(class2)
	var1=np.var(class1)
	var2=np.var(class2)

	fisher=(m1-m2)**2/(var1+var2)**2

	return fisher

def information_gain_selection(X,y,n_features):
	"""
	Computes the Information Gain of each feature and selects the top ranking features

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	n_features -- n best ranked features
	"""

	if verbose:
		print '\nPerforming Feature Selection based on the Information Gain ...'

	feature_scores=[]
	for i in xrange(len(X[0])):
		feature_scores+=[(information_gain(X,y,i),i)]							#compute scores

	feature_scores.sort(reverse=True)
	scores,feature_indexes=split_tuples(feature_scores)							#split into score and indexes lists

	return X[:,feature_indexes[0:n_features]],feature_indexes[0:n_features]		#return selected features and original index features


def gain_ratio_selection(X,y,n_features):
	"""
	Computes the Gain Ratio of each feature and selects the top ranking features

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	n_features -- n best ranked features
	"""

	if verbose:
		print '\nPerforming Feature Selection based on the Gain Ratio ...'

	feature_scores=[]
	for i in xrange(len(X[0])):
		feature_scores+=[(gain_ratio(X,y,i),i)]									#compute scores

	feature_scores.sort(reverse=True)
	scores,feature_indexes=split_tuples(feature_scores)							#split into score and indexes lists

	return X[:,feature_indexes[0:n_features]],feature_indexes[0:n_features]		#return selected features and original index features

def chi_squared_selection(X,y,n_features):
	"""
	Computes the Chi-squared statistic of each feature and selects the top ranking features

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	n_features -- n best ranked features
	"""

	if verbose:
		print '\nPerforming Feature Selection based on the Chi2 test ...'
	feature_scores=[]
	for i in xrange(len(X[0])):
		chi2_stat,p=chi2_feature_test(X,y,i)
		feature_scores+=[(chi2_stat,i)]											#compute scores

	feature_scores.sort(reverse=True)
	scores,feature_indexes=split_tuples(feature_scores)							#split into score and indexes lists

	return X[:,feature_indexes[0:n_features]],feature_indexes[0:n_features]		#return selected features and original index features

def kruskal_wallis_selection(X,y,n_features):
	"""
	Computes the Kruskal-Wallis statistic of each feature and selects the top ranking features

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	n_features -- n best ranked features
	"""

	if verbose:
		print '\nPerforming Feature Selection based on the Kruskal-Wallis test ...'
	feature_scores=[]
	for i in xrange(len(X[0])):
		H_kw,kw_p_val=kw_feature_test(X,y,i)
		feature_scores+=[(H_kw,i)]												#compute scores

	feature_scores.sort(reverse=True)
	scores,feature_indexes=split_tuples(feature_scores)							#split into score and indexes lists

	return X[:,feature_indexes[0:n_features]],feature_indexes[0:n_features]		#return selected features and original index features


def fisher_score_selection(X,y,n_features):
	"""
	Computes the Fisher Score of each feature and selects the top ranking features

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	n_features -- n best ranked features
	"""

	if verbose:
		print '\nPerforming Feature Selection based on the Fisher score ...'

	feature_scores=[]
	for i in xrange(len(X[0])):
		feature_scores+=[(fisher_score(X,y,i),i)]								#compute scores

	feature_scores.sort(reverse=True)
	scores,feature_indexes=split_tuples(feature_scores)							#split into score and indexes listss

	return X[:,feature_indexes[0:n_features]],feature_indexes[0:n_features]		#return selected features and original index features

def pearson_between_feature(X,threshold):
	"""
	Computes the Pearson Correlation between each feature and drops the higlhy correlated features

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	threshold -- Threshold value used to decide which features to keep (above the threshold)
	"""

	if verbose:
		print '\nPerforming Feature Selection based on the correlation between each feature ...'

	feature_indexes=[]
	for i in xrange(len(X[0])):
		correlated=False
		for j in xrange(i+1,len(X[0])):
			if abs(stats.pearsonr(X[:,i],X[:,j])[0])>threshold:
				correlated=True
		if not correlated:
			feature_indexes+=[i]


	if len(feature_indexes)!=0:
		return X[:,feature_indexes],feature_indexes		#return selected features and original index features
	else:
		return X,feature_indexes

def pearson_between_feature_class(X,y,threshold):
	"""
	Computes the Pearson Correlation between each feature and the target class and keeps the higlhy correlated features-class

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	threshold -- Threshold value used to decide which features to keep (above the threshold)
	"""

	if verbose:
		print '\nPerforming Feature Selection based on the correlation between each feature and class ...'

	feature_indexes=[]
	for i in xrange(len(X[0])):
		if abs(stats.pearsonr(X[:,i],y)[0])>threshold:
			feature_indexes+=[i]

	if len(feature_indexes)!=0:
		return X[:,feature_indexes],feature_indexes		#return selected features and original index features
	else:
		return X,feature_indexes

def mrmr_selection(X,features,y,n_features,selection_method):
	"""
	Performs the Minimum Redundancy & Maximum Relevance selection method

	Keyword arguments:
	X -- The feature vectors
	features -- feature names
	y -- The target vector
	threshold -- threshold to discretize the values
	selection_method -- either 'MID' (Additive) or 'MIQ' (Multiplicative)
	"""
	if verbose:
		print '\nPerforming Feature Selection based on the mRMR selection method...'
	mrmr_result = mrmr(X, features[:-1], y, threshold=1,nFeats=n_features,mrmrexe='peng_mrmr/mrmr')
	feature_indexes=mrmr_result['mRMR']['Fea']
	return X[:,feature_indexes],feature_indexes

def auc_selection(X,y,n_features):
	"""
	Computes the Area Under the Curve achieved by each feature and selects the top ranking features

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	n_features -- n best ranked features
	"""

	if verbose:
		print '\nPerforming Feature Selection based on the AUC from the ROC curves...'

	feature_scores=[]
	for i in xrange(len(X[0])):
		feature_scores+=[(roc_auc_score(y,X[:,i]),i)]							#compute scores

	feature_scores.sort(reverse=True)
	scores,feature_indexes=split_tuples(feature_scores)							#split into score and indexes lists

	return X[:,feature_indexes[0:n_features]],feature_indexes[0:n_features]		#return selected features and original index features

def rfe_selection(X,y,n_features):
	"""
	Performs the Recursive Feature Elimination method and selects the top ranking features

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	n_features -- n best ranked features
	"""

	if verbose:
		print '\nPerforming Feature Selection based on the Recursive Feature Elimination method ...'

	clf=RandomForestClassifierWithCoef(n_estimators=10,n_jobs=-1)
	fs= RFE(clf, n_features, step=1)
	fs= fs.fit(X,y)
	ranks=fs.ranking_

	feature_indexes=[]
	for i in xrange(len(ranks)):
		if ranks[i]==1:
			feature_indexes+=[i]

	return X[:,feature_indexes[0:n_features]],feature_indexes[0:n_features]		#return selected features and original index features

def sfs_selection(X,y,n_features,forward):
	"""
	Performs the Sequential Forward/Backward Selection method and selects the top ranking features

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	n_features -- n best ranked features
	"""

	if verbose:
		print '\nPerforming Feature Selection based on the Sequential Feature Selection method ...'

	clf=RandomForestClassifierWithCoef(n_estimators=5,n_jobs=-1)
	sfs = SFS(clf,k_features=n_features,forward=forward,scoring='accuracy',cv=0,n_jobs=-1, print_progress=True,)
	sfs = sfs.fit(X, y)

	feature_indexes=sfs.k_feature_idx_
	return X[:,feature_indexes[0:n_features]],feature_indexes[0:n_features]		#return selected features and original index features



def pca_selection(X,n_components):
	"""
	Computes the Principal Components and keeps the most significant ones

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	n_components -- Number of principal components to keep
	"""

	if verbose:
		print '\nPerforming Principal Component Analysis ...'

	pca = PCA(n_components = n_components)
	principal_components = pca.fit_transform(X)
	return principal_components

def lda_selection(X,y,n_components):
	"""
	Performs the Fisher's Linear Discrimination Analysis keeps the most discriminative features

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	n_components -- Number of features to keep
	"""

	if verbose:
		print '\nPerforming Linear Discrimination Analysis ...'

	lda = LDA(n_components = n_components,solver='eigen')
	discriminative_attributes = lda.fit(X, y).transform(X)
	return discriminative_attributes

#Random Forest Classifier with an additional attribute coef_, in order to be usable by the Recursive Feature Elimination method
class RandomForestClassifierWithCoef(RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_
