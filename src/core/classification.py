import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances

class MinimumDistanceClassifier(BaseEstimator, ClassifierMixin):

	def __init__(self,distance_metric="euclidean",n_classes=2,outcomes=[0,1]):
		"""
		Initializes the classifier

		Keyword arguments:
		distance_metric -- The distance metric to be used when computing distances between vectors
		n_classes -- The number of classes
		outcomes -- The possible labels
		"""
		
		self.distance_metric=distance_metric
		self.n_classes=n_classes
		self.outcomes=outcomes

	def fit(self, X, y):
		"""
		Computes the median feature vectors to be used as prototypes

		Keyword arguments:
		X -- The feature vectors
		y -- The target vectors
		"""

		#input validation
		assert(type(self.distance_metric) == str)
		assert(type(self.n_classes) == int)
		assert(type(self.outcomes) == list)
		assert(len(self.outcomes)==self.n_classes)

		#Compute prototypes
		self.M_=np.empty([self.n_classes,len(X[0])])	#init random prototypes
		m_index=0
		for c in self.outcomes:
			class_samples=X[y==c,:]						#grab samples from class i

			mi=np.mean(class_samples,axis=0)			#compute the mean feature vector
			self.M_[m_index]=mi
			m_index+=1

		return self
	def predict(self, X):
		"""
		Classify the input data assigning the label of the nearest prototype

		Keyword arguments:
		X -- The feature vectors
		"""
		classification=np.zeros(len(X))

		if self.distance_metric=="euclidean":
			distances=pairwise_distances(X, self.M_,self.distance_metric)									#compute distances to the prototypes (template matching)
		if self.distance_metric=="minkowski":
			distances=pairwise_distances(X, self.M_,self.distance_metric)	
		elif self.distance_metric=="manhattan":
			distances=pairwise_distances(X, self.M_,self.distance_metric)
		elif self.distance_metric=="mahalanobis":
			distances=pairwise_distances(X, self.M_,self.distance_metric)
		else:
			distances=pairwise_distances(X, self.M_,"euclidean")

		for i in xrange(len(X)):
			classification[i]=self.outcomes[distances[i].tolist().index(min(distances[i]))]					#choose the class belonging to nearest prototype distance

		return classification

	def score(self,X, y):
		"""
		Computes the number of True Positives achieved by classification

		Keyword arguments:
		X -- The feature vectors
		y -- The target vectors
		"""
		predictions=self.predict(X)

		count=0
		for i in xrange(len(predictions)):
			if predictions[i]==y[i]:
				count+=1

		return count
