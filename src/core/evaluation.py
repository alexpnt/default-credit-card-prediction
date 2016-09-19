from sklearn.metrics import accuracy_score,average_precision_score,f1_score,precision_score,recall_score
from sklearn.metrics import confusion_matrix,roc_curve,roc_auc_score,auc,precision_recall_curve
from sklearn.cross_validation import StratifiedKFold
import numpy as np
from numpy import interp
import matplotlib.pyplot as plt
from utils import save_fig

def get_accuracy(y_gold_standard,y_predicted):
	"""
	Computes the accuracy classification score.

	Keyword arguments:
	y_gold_standard -- Expected labels
	y_predicted -- Predicted labels
	"""
	return accuracy_score(y_gold_standard, y_predicted)

def get_average_precision(y_gold_standard,y_predicted):
	"""
	Computes the average precision score. Also known as the area under the precision-recall curve.

	Keyword arguments:
	y_gold_standard -- Expected labels
	y_predicted -- Predicted labels
	"""

	return average_precision_score(y_gold_standard, y_predicted)

def get_f1_score(y_gold_standard,y_predicted):
	"""
	Computes the F1 score.

	Keyword arguments:
	y_gold_standard -- Expected labels
	y_predicted -- Predicted labels
	"""

	return f1_score(y_gold_standard, y_predicted)

def get_precision_score(y_gold_standard,y_predicted):
	"""
	Computes the precision score.

	Keyword arguments:
	y_gold_standard -- Expected labels
	y_predicted -- Predicted labels
	"""

	return precision_score(y_gold_standard, y_predicted)

def get_recall_score(y_gold_standard,y_predicted):
	"""
	Computes the recall score. Also known as the area under the precision-recall curve.

	Keyword arguments:
	y_gold_standard -- Expected labels.
	y_predicted -- Predicted labels
	"""

	return recall_score(y_gold_standard, y_predicted)

def get_confusion_matrix(y_gold_standard,y_predicted):
	"""
	Computes the confusion matrix.

	Keyword arguments:
	y_gold_standard -- Expected labels.
	y_predicted -- Predicted labels
	"""

	return confusion_matrix(y_gold_standard, y_predicted)

def get_roc_curve(y_gold_standard,y_predicted):
	"""
	Computes the Receiver Operating Characteristic.

	Keyword arguments:
	y_gold_standard -- Expected labels.
	y_predicted -- Predicted labels
	"""

	return roc_curve(y_gold_standard, y_predicted)

def get_auc_score(y_gold_standard,y_predicted):
	"""
	Computes the Area Under the Curve (AUC).

	Keyword arguments:
	y_gold_standard -- Expected labels.
	y_predicted -- Predicted labels
	"""

	return roc_auc_score(y_gold_standard, y_predicted)

def get_precision_recall_curve(y_gold_standard,y_predicted):
	"""
	Computes the precision-recall curve.

	Keyword arguments:
	y_gold_standard -- Expected labels.
	y_predicted -- Predicted labels
	"""

	return precision_recall_curve(y_gold_standard, y_predicted)

def visualize_k_fold_roc_plot(X,y_gold,classifier,K):
	"""
	Visualizes K ROC curves created from K-fold cross validation and the mean ROC curve.

	Keyword arguments:
	X -- The feature vectors
	y_gold_standard -- Expected labels.
	classifier -- The classifier to be used
	K -- Number of folds to perform
	"""

	cross_validation = StratifiedKFold(y_gold, n_folds=K)

	mean_true_positive_rate = 0.0
	mean_false_positive_rate = 0.0

	for i, (train, test) in enumerate(cross_validation):
		#classify
		classifier.fit(X[train], y_gold[train])
		y_predicted=classifier.predict(X[test])

		#compute ROC
		false_positive_rate, true_positive_rate, thresholds = roc_curve(y_gold[test], y_predicted)
		roc_auc = auc(false_positive_rate, true_positive_rate)
		plt.plot(false_positive_rate, true_positive_rate, linewidth=1, label='ROC fold %d (area = %0.2f)' % (i+1, roc_auc))

		#save means
		mean_true_positive_rate += true_positive_rate
		mean_false_positive_rate += false_positive_rate

	#compute final mean
	mean_true_positive_rate /= len(cross_validation)
	mean_false_positive_rate /= len(cross_validation)
	mean_auc = auc(mean_false_positive_rate, mean_true_positive_rate)
	plt.plot(mean_false_positive_rate, mean_true_positive_rate, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, linewidth=2)


	plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random Classifier')

	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic (ROC) Curve')
	plt.legend(loc="lower right")
	# plt.show()

	#save fig
	output_dir='img'
	save_fig(output_dir,'{}/roc.png'.format(output_dir))

	plt.close()

def visualize_k_fold_precision_recall_plot(X,y_gold,classifier,K):
	"""
	Visualizes K Average Precision-Recall curves created from K-fold cross validation and the mean Precision-Recall curve.

	Keyword arguments:
	X -- The feature vectors
	y_gold_standard -- Expected labels.
	classifier -- The classifier to be used
	K -- Number of folds to perform
	"""

	cross_validation = StratifiedKFold(y_gold, n_folds=K)

	mean_precision = 0.0
	mean_recall= 0.0
	avg_mean_precision_recall=0.0

	for i, (train, test) in enumerate(cross_validation):
		#classify
		classifier.fit(X[train], y_gold[train])
		y_predicted=classifier.predict(X[test])

		#compute Precision-Recall
		precision, recall, thresholds = precision_recall_curve(y_gold[test].ravel(),y_predicted.ravel())
		average_precision = average_precision_score(y_gold[test], y_predicted)
		plt.plot(recall, precision,label='Precision-recall fold %d (area = %0.2f)' % (i+1, average_precision) )

		#save means
		mean_precision += precision
		mean_recall += recall
		avg_mean_precision_recall+=average_precision

    #compute final mean
	mean_precision /= len(cross_validation)
	mean_recall /= len(cross_validation)
	avg_mean_precision_recall /= len(cross_validation)
	plt.plot(mean_recall,mean_precision, 'k--',label='Mean Precision-Recall (area = %0.2f)' % avg_mean_precision_recall, linewidth=2)

	plt.plot([0,1], [0.5, 0.5], '--', color=(0.6, 0.6, 0.6), label='Random Classifier')

	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall Curve')
	plt.legend(loc="lower left")
	# plt.show()

	#save fig
	output_dir='img'
	save_fig(output_dir,'{}/pr_curve.png'.format(output_dir))

	plt.close()

def get_k_fold_metrics(X,y_gold,classifier,K):
	"""
	Computes the final confusion matrix (sum of all confusion matrixes obtained during the K-fold Cross Validation)

	Keyword arguments:
	X -- The feature vectors
	y_gold_standard -- Expected labels.
	classifier -- The classifier to be used
	K -- Number of folds to perform
	"""

	cross_validation = StratifiedKFold(y_gold, n_folds=K)

	final_confusion_matrix=np.zeros(shape=(2,2))
	accuracy=[]
	precision=[]
	recall=[]
	f1=[]
	avg_precision_recall=[]
	auc=[]
	for i, (train, test) in enumerate(cross_validation):
		#classify
		classifier.fit(X[train], y_gold[train])
		y_predicted=classifier.predict(X[test])

		#save metrics
		final_confusion_matrix+=get_confusion_matrix(y_gold[test],y_predicted)
		accuracy+=[get_accuracy(y_gold[test],y_predicted)]
		precision+=[get_precision_score(y_gold[test],y_predicted)]
		recall+=[get_recall_score(y_gold[test],y_predicted)]
		f1+=[get_f1_score(y_gold[test],y_predicted)]
		avg_precision_recall+=[get_average_precision(y_gold[test],y_predicted)]
		auc+=[get_auc_score(y_gold[test],y_predicted)]

	#compute stds
	accuracy_std=np.std(accuracy)
	precision_std=np.std(precision)
	recall_std=np.std(recall)
	f1_std=np.std(f1)
	avg_precision_recall_std=np.std(avg_precision_recall)
	auc_std=np.std(auc)

	#compute means
	accuracy_mean=np.mean(accuracy)
	precision_mean=np.mean(precision)
	recall_mean=np.mean(recall)
	f1_mean=np.mean(f1)
	avg_precision_recall_mean=np.mean(avg_precision_recall)
	auc_mean=np.mean(auc)

	return final_confusion_matrix,\
		   (accuracy_mean,accuracy_std),\
		   (precision_mean,precision_std),\
		   (recall_mean,recall_std),\
		   (f1_mean,f1_std),\
		   (avg_precision_recall_mean,avg_precision_recall_std)\
		,(auc_mean,auc_std)

