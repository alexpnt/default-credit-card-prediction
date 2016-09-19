#!/usr/bin/env python2
# -*- coding: utf-8 -*

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sea
import pandas as pd
from utils import mkdir_p
from feature_selection import kolmogorov_smirnov_two_sample_test
sea.set()

from utils import save_fig

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def visualize_pca2D(X,y):
	"""
	Visualize the first two principal components

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	"""
	pca = PCA(n_components = 2)
	principal_components = pca.fit_transform(X)

	palette = sea.color_palette()
	plt.scatter(principal_components[y==0, 0], principal_components[y==0, 1], marker='s',color='green',label="Paid", alpha=0.5,edgecolor='#262626', facecolor=palette[1], linewidth=0.15)
	plt.scatter(principal_components[y==1, 0], principal_components[y==1, 1], marker='^',color='red',label="Default", alpha=0.5,edgecolor='#262626''', facecolor=palette[2], linewidth=0.15)

	leg = plt.legend(loc='upper right', fancybox=True)
	leg.get_frame().set_alpha(0.5)
	plt.title("Two-Dimensional Principal Component Analysis")
	plt.tight_layout

	#save fig
	output_dir='img'
	save_fig(output_dir,'{}/pca2D.png'.format(output_dir))

def visualize_pca3D(X,y):
	"""
	Visualize the first three principal components

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	"""
	pca = PCA(n_components = 3)
	principal_components = pca.fit_transform(X)

	fig = pylab.figure()
	ax = Axes3D(fig)
	# azm=30
	# ele=30
	# ax.view_init(azim=azm,elev=ele)

	palette = sea.color_palette()
	ax.scatter(principal_components[y==0, 0], principal_components[y==0, 1], principal_components[y==0, 2], label="Paid", alpha=0.5, 
	            edgecolor='#262626', c=palette[1], linewidth=0.15)
	ax.scatter(principal_components[y==1, 0], principal_components[y==1, 1], principal_components[y==1, 2],label="Default", alpha=0.5, 
	            edgecolor='#262626''', c=palette[2], linewidth=0.15)

	ax.legend()
	plt.show()

def visualize_lda2D(X,y):
	"""
	Visualize the separation between classes using the two most discriminant features

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	"""
	labels=['Paid','Default']
	lda = LDA(n_components = 2,solver='eigen')
	# lda = LDA(n_components = 2)
	discriminative_attributes = lda.fit(X, y).transform(X)

	palette = sea.color_palette()
	# plt.plot(discriminative_attributes[:,0][y==0],'sg',label="Paid", alpha=0.5)
	# plt.plot(discriminative_attributes[:,0][y==1],'^r',label="Default", alpha=0.5)
	plt.scatter(discriminative_attributes[:,0][y==0],discriminative_attributes[:,1][y==0],marker='s',color='green',label="Paid", alpha=0.5)
	plt.scatter(discriminative_attributes[:,0][y==1],discriminative_attributes[:,1][y==1],marker='^',color='red',label="Default", alpha=0.5)
	plt.xlabel('First Linear Discriminant')
	plt.ylabel('Second Linear Discriminant')

	leg = plt.legend(loc='upper right', fancybox=True)
	leg.get_frame().set_alpha(0.5)
	plt.title("Linear Discriminant Analysis")
	plt.tight_layout

	#save fig
	output_dir='img'
	save_fig(output_dir,'{}/lda.png'.format(output_dir))

def visualize_feature_hist_dist(X,y,selected_feature,features,normalize=False):
	"""
	Visualize the histogram distribution of a feature

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	selected_feature -- The desired feature to obtain the histogram
	features -- Vector of feature names (X1 to XN)
	normalize -- Whether to normalize the histogram (Divide by total)
	"""

	#create data
	joint_data=np.column_stack((X,y))
	column_names=features

	#create dataframe
	df=pd.DataFrame(data=joint_data,columns=column_names)
	palette = sea.hls_palette()

	#find number of unique values (groups)
	unique_values=pd.unique(df[[selected_feature]].values.ravel())
	unique_values=map(int, unique_values)
	unique_values.sort()
	n_groups=len(unique_values)

	fig, ax = plt.subplots()
	index = np.arange(n_groups)
	bar_width = 0.35
	opacity = 0.4

	#find values belonging to the positive class and values belonging to the negative class
	positive_class_index=df[df[features[-1]] == 1].index.tolist()
	negative_class_index=df[df[features[-1]] != 1].index.tolist()

	positive_values=df[[selected_feature]].loc[positive_class_index].values.ravel()
	positive_values=map(int, positive_values)

	negative_values=df[[selected_feature]].loc[negative_class_index].values.ravel()
	negative_values=map(int, negative_values)

	#normalize data (divide by total)
	n_positive_labels=n_negative_labels=1
	if normalize==True:
		n_positive_labels=len(y[y==1])
		n_negative_labels=len(y[y!=1])

	#count
	positive_counts=[0]*len(index)
	negative_counts=[0]*len(index)
	for v in xrange(len(unique_values)):
		positive_counts[v]=positive_values.count(v)/n_positive_labels
		negative_counts[v]=negative_values.count(v)/n_negative_labels

	#plot
	plt.bar(index, positive_counts, bar_width,alpha=opacity,color='b',label='Default')			#class 1
	plt.bar(index+bar_width, negative_counts, bar_width,alpha=opacity,color='r',label='Paid')	#class 0

	plt.xlabel(selected_feature)
	plt.ylabel('Frequency')
	if normalize:
		plt.ylabel('Proportion')
	plt.title("Normalized Histogram Distribution of the feature '"+selected_feature+"' grouped by class")
	plt.xticks(index + bar_width, map(str, unique_values) )
	plt.legend()
	plt.tight_layout()

	# plt.show()

	#save fig
	output_dir = "img"
	save_fig(output_dir,'{}/{}_hist_dist.png'.format(output_dir,selected_feature))


def visualize_hist_pairplot(X,y,selected_feature1,selected_feature2,features,diag_kind):
	"""
	Visualize the pairwise relationships (Histograms and Density Funcions) between classes and respective attributes

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	selected_feature1 - First feature
	selected_feature1 - Second feature
	diag_kind -- Type of plot in the diagonal (Histogram or Density Function)
	"""

	#create data
	joint_data=np.column_stack((X,y))
	column_names=features

	#create dataframe
	df=pd.DataFrame(data=joint_data,columns=column_names)

	#plot
	palette = sea.hls_palette()
	splot=sea.pairplot(df, hue="Y", palette={0:palette[2],1:palette[0]},vars=[selected_feature1,selected_feature2],diag_kind=diag_kind)
	splot.fig.suptitle('Pairwise relationship: '+selected_feature1+" vs "+selected_feature2)
	splot.set(xticklabels=[])
	# plt.subplots_adjust(right=0.94, top=0.94)

	#save fig
	output_dir = "img"
	save_fig(output_dir,'{}/{}_{}_hist_pairplot.png'.format(output_dir,selected_feature1,selected_feature2))
	# plt.show()


def visualize_hist_pairplots(X,y):
	"""
	Visualize the pairwise relationships (Histograms and Density Funcions) between classes and respective attributes

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	"""

	joint_data=np.column_stack((X,y))
	df=pd.DataFrame(data=joint_data,columns=["Credit","Gender","Education","Marital Status","Age","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18","X19","X20","X21","X22","X23","Default"])

	palette = sea.hls_palette()

	#histograms	
	splot=sea.pairplot(df, hue="Default", palette={0:palette[2],1:palette[0]},vars=["Credit","Gender","Education","Marital Status","Age"])
	splot.fig.suptitle('Histograms Distributions and Scatter Plots: Credit, Gender, Education, Marital Status and Age')
	splot.set(xticklabels=[])
	plt.subplots_adjust(right=0.94, top=0.94)
	plt.show()

	splot=sea.pairplot(df, hue="Default", palette={0:palette[2],1:palette[0]},vars=["X6","X7","X8","X9","X10","X11"])
	splot.fig.suptitle('Histograms Distributions and Scatter Plots: History of Payment')
	splot.set(xticklabels=[])
	plt.subplots_adjust(right=0.94, top=0.94)
	plt.show()

	splot=sea.pairplot(df, hue="Default", palette={0:palette[2],1:palette[0]},vars=["X12","X13","X14","X15","X16","X17"])
	splot.fig.suptitle('Histograms Distributions and Scatter Plots: Amount of Bill Statements')
	splot.set(xticklabels=[])
	plt.subplots_adjust(right=0.94, top=0.94)
	plt.show()

	splot=sea.pairplot(df, hue="Default", palette={0:palette[2],1:palette[0]},vars=["X18","X19","X20","X21","X22","X23"])
	splot.fig.suptitle('Histograms Distributions and Scatter Plots: Amount of Previous Payments')
	splot.set(xticklabels=[])
	plt.subplots_adjust(right=0.94, top=0.94)
	plt.show()

	#kdes
	splot=sea.pairplot(df, hue="Default", palette={0:palette[2],1:palette[0]},diag_kind="kde",vars=["Credit","Gender","Education","Marital Status","Age"])
	splot.fig.suptitle('Univariate Kernel Density Estimations and Scatter Plots: Credit, Gender, Education, Marital Status and Age')
	splot.set(xticklabels=[])
	plt.subplots_adjust(right=0.94, top=0.94)
	plt.show()

	splot=sea.pairplot(df, hue="Default", palette={0:palette[2],1:palette[0]},diag_kind="kde",vars=["X6","X7","X8","X9","X10","X11"])
	splot.fig.suptitle('Univariate Kernel Density Estimations and Scatter Plots: History of Payment')
	splot.set(xticklabels=[])
	plt.subplots_adjust(right=0.94, top=0.94)
	plt.show()

	splot=sea.pairplot(df, hue="Default", palette={0:palette[2],1:palette[0]},diag_kind="kde",vars=["X12","X13","X14","X15","X16","X17"])
	splot.fig.suptitle('Univariate Kernel Density Estimations and Scatter Plots: Amount of Bill Statements')
	splot.set(xticklabels=[])
	plt.subplots_adjust(right=0.94, top=0.94)
	plt.show()

	splot=sea.pairplot(df, hue="Default", palette={0:palette[2],1:palette[0]},diag_kind="kde",vars=["X18","X19","X20","X21","X22","X23"])
	splot.fig.suptitle('Univariate Kernel Density Estimations and Scatter Plots: Amount of Previous Payments')
	splot.set(xticklabels=[])
	plt.subplots_adjust(right=0.94, top=0.94)
	plt.show()

def visualize_feature_boxplot(X,y,selected_feature,features):
	"""
	Visualize the boxplot of a feature

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	selected_feature -- The desired feature to obtain the histogram
	features -- Vector of feature names (X1 to XN)
	"""

	#create data
	joint_data=np.column_stack((X,y))
	column_names=features

	#create dataframe
	df=pd.DataFrame(data=joint_data,columns=column_names)

	# palette = sea.hls_palette()
	splot=sea.boxplot(data=df,x='Y',y=selected_feature,hue="Y",palette="husl")
	plt.title('BoxPlot Distribution of '+selected_feature)

	#save fig
	output_dir = "img"
	save_fig(output_dir,'{}/{}_boxplot.png'.format(output_dir,selected_feature))
	# plt.show()



def visualize_boxplots(X,y):
	"""
	Visualize the boxplots of the features

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	"""

	credit=X[:,0:1]
	df=pd.DataFrame(data=credit,columns=["Credit"])
	splot=sea.boxplot(data=df, orient="h",palette="husl")
	plt.title('BoxPlot Distribution of Credit')
	plt.show()

	one_to_four_columns=X[:,1:4]
	df=pd.DataFrame(data=one_to_four_columns,columns=["Gender","Education","Marital Status"])
	splot=sea.boxplot(data=df, orient="h",palette="husl")
	plt.title('BoxPlot Distribution of Features: Gender, Education and Marital Status')
	plt.show()

	age=X[:,4:5]
	df=pd.DataFrame(data=age,columns=["Age"])
	splot=sea.boxplot(data=df, orient="h",palette="husl")
	plt.title('BoxPlot Distribution of Age')
	plt.show()

	x6_to_x11=X[:,5:11]
	df=pd.DataFrame(data=x6_to_x11,columns=["X6","X7","X8","X9","X10","X11"])
	splot=sea.boxplot(data=df, orient="h",palette="husl")
	plt.title('BoxPlot Distribution of Features: History of Payment')
	plt.show()

	x12_to_x17=X[:,11:17]
	df=pd.DataFrame(data=x12_to_x17,columns=["X12","X13","X14","X15","X16","X17"])
	splot=sea.boxplot(data=df, orient="h",palette="husl")
	plt.title('BoxPlot Distribution of Features: Amount of Bill Statements')
	plt.show()

	x18_to_x23=X[:,17:23]
	df=pd.DataFrame(data=x12_to_x17,columns=["X18","X19","X20","X21","X22","X23"])
	splot=sea.boxplot(data=df, orient="h",palette="husl")
	plt.title('BoxPlot Distribution of Features: Amount of Previous Payments')
	plt.show()

def feature_cdf(X,y,selected_feature):
	"""
	Plot the empirical/stand cumulative density function of the given feature

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	selected_feature -- The desired feature to obtain the histogram
	"""

	#Standard Normal Cumulative Density Function
	N = len(X)
	Normal = np.random.normal(size = N)
	histogram,bin_edges = np.histogram(Normal, bins = N, normed = True )
	dx = bin_edges[1] - bin_edges[0]
	G = np.cumsum(histogram)*dx

	#Empirical Cumulative Density Functions
	feature_index=int(selected_feature[1:])-1
	X_k = np.sort(X[:,feature_index])					#feature vector sorted
	ECDF_k = np.array(range(N))/float(N)				#Empirical Cumulative Function F, steps of 1/N

	#Kolmogorov-Smirnov Test
	result=kolmogorov_smirnov_two_sample_test(G,ECDF_k)
	ks_statistic=result[0]
	p_value=result[1]

	plt.plot(bin_edges[1:], G, label="Standard Normal Cumulative Density Funcion")
	plt.plot(X_k, ECDF_k,label="Empirical Cumulative Density Function")
	plt.suptitle("Empirirical vs Standard Normal Cumulative Distribution of "+selected_feature+" Feature\nKolmogorov-Smirnov Statistic="+str(ks_statistic))
	plt.xlabel(selected_feature)
	plt.legend(loc='center right')

	# plt.show()

	#save fig
	output_dir = "img"
	save_fig(output_dir,'{}/{}_cdf.png'.format(output_dir,selected_feature))



def cdf(X,y):
	"""
	Plot the empirical/stand cumulative density function of the features 

	Keyword arguments:
	X -- The feature vectors
	y -- The target vector
	"""

	#Standard Normal Cumulative Density Function 
	N = len(X)
	Normal = np.random.normal(size = N)
	histogram,bin_edges = np.histogram(Normal, bins = 10, normed = True )
	dx = bin_edges[1] - bin_edges[0]
	G = np.cumsum(histogram)*dx

	#Empirical Cumulative Functions

	#Credit
	X0 = np.sort(X[:,0])
	F0 = np.array(range(N))/float(N)

	plt.plot(bin_edges[1:], G, label="Standard Normal Cumulative Density Funcion")
	plt.plot(X0, F0,label="Empirical Cumulative Density Function")
	plt.suptitle("Empirirical vs Standard Normal Cumulative Distribution of Credit Feature")
	plt.xlabel("Credit")
	plt.legend(loc='center right')
	plt.show()
	

	x_axis=["Gender","Education","Marital Status"]
	for i in xrange(3):
		Xi = np.sort(X[:,i+1])
		Fi = np.array(range(N))/float(N)

		plt.subplot(1,3,i+1)
		plt.plot(bin_edges[1:], G, label="Standard Normal CDF")
		plt.plot(Xi, Fi, label="Empirical CDF")
		plt.xlabel(x_axis[i])
		plt.legend(loc='upper left')
	plt.suptitle("Empirirical vs Standard Normal Cumulative Distribution of Features: Gender, Education and Marital Status")
	plt.show()

	#Age
	X0 = np.sort(X[:,4])
	F0 = np.array(range(N))/float(N)

	plt.plot(bin_edges[1:], G, label="Standard Normal Cumulative Density Funcion")
	plt.plot(X0, F0,label="Empirical Cumulative Density Function")
	plt.suptitle("Empirirical vs Standard Normal Cumulative Distribution of Age Feature")
	plt.xlabel("Age")
	plt.legend(loc='center right')
	plt.show()

	x_axis=["X6","X7","X8","X9","X10","X11"]
	for i in xrange(5,11):
		Xi = np.sort(X[:,i])
		Fi = np.array(range(N))/float(N)

		plt.subplot(3,2,i-4)
		plt.plot(bin_edges[1:], G, label="Standard Normal CDF")
		plt.plot(Xi, Fi, label="Empirical CDF")
		plt.xlabel(x_axis[i-5])
		plt.legend(loc='upper left')
	plt.suptitle("Empirirical vs Standard Normal Cumulative Distribution of Features: History of Payment")
	plt.show()

	x_axis=["X12","X13","X14","X15","X16","X17"]
	for i in xrange(11,17):
		Xi = np.sort(X[:,i])
		Fi = np.array(range(N))/float(N)

		plt.subplot(3,2,i-10)
		plt.plot(bin_edges[1:], G, label="Standard Normal CDF")
		plt.plot(Xi, Fi, label="Empirical CDF")
		plt.xlabel(x_axis[i-11])
		plt.legend(loc='lower right')
	plt.suptitle("Empirirical vs Standard Normal Cumulative Distribution of Features: Amount of Bill Statements")
	plt.show()

	x_axis=["X18","X19","X20","X21","X22","X23"]
	for i in xrange(18,23):
		Xi = np.sort(X[:,i])
		Fi = np.array(range(N))/float(N)

		plt.subplot(3,2,i-17)
		plt.plot(bin_edges[1:], G, label="Standard Normal CDF")
		plt.plot(Xi, Fi, label="Empirical CDF")
		plt.xlabel(x_axis[i-18])
		plt.legend(loc='lower right')
	plt.suptitle("Empirirical vs Standard Normal Cumulative Distribution of Features: Amount of Previous Payments")
	plt.show()






