Default-Credit-Card-Prediction
===================================
Machine Learning Project for predicting Credit Card Defaults.

#### Features: ####
* Dataset Loading
* Feature Assessment/Visualization: 
	* Normalized Histogram Distribution
	* Box Plots
	* Pairwise Relationships
	* Empirical Cumulative/Standard Density Functions
	* Pearson Correlation
	* 2D PCA
	* 2D LDA
* Preprocessing:
	* Standardization
	* Scaling
	* Normalization
	* Dataset Balancing
* Feature Selection:
	* (Filter) Information Gain
	* (Filter) Gain Ratio
	* (Filter) Chi-squared Test
	* (Filter) Kruskal-Wallis Test
	* (Filter) Fisher Score
	* (Filter) Pearson Correlation (Feature-Feature, Feature-Class)
	* (Filter) mRMR
	* (Filter) Area Under the Curve (AUC)
	* (Wrapper) Sequential Forward/Backward Selection
	* (Wrapper) Recursive Feature Elimination
* Feature Reduction: 
	* Principal Component Analysis (PCA)
	* Fisher's Linear Discriminant Analysis (LDA)
* Classification: 
	* Minimum Distance Classifier
	* k-Nearest-Neighbors (kNN)
	* Naive Bayes
	* Support Vector Machines (SVM)
	* Decision Tree (CART)
	* Random Forest
* Evaluation: 
	* Stratified K-folds Cross Validation
	* Receiver Operating Characteristic (ROC) Curves
	* Precision-Recall Curves


####Requirements:####
* python 2.x
* scikit-learn (http://scikit-learn.org/stable)
* sciPy (http://www.scipy.org)
* numPy (http://www.numpy.org)
* matplotlib (http://matplotlib.org)
* mlxtend (http://rasbt.github.io/mlxtend)
* UnbalancedDataset (https://github.com/shubhabrataroy/UnbalancedDataset)
* pandas (http://pandas.pydata.org)
* seaborn (https://stanford.edu/~mwaskom/software/seaborn)
* PyQT5 (https://pypi.python.org/pypi/PyQt5)

####Usage:####

    usage: python default_predictor.py
                        
####Examples:####

![eg](https://raw.githubusercontent.com/AlexPnt/Default-Credit-Card-Prediction/master/figures/gui/Loading.png)

![eg](https://raw.githubusercontent.com/AlexPnt/Default-Credit-Card-Prediction/master/figures/gui/feature-inspection-hist.png)

![eg](https://raw.githubusercontent.com/AlexPnt/Default-Credit-Card-Prediction/master/figures/gui/feature-inspection-boxplot.png)

![eg](https://raw.githubusercontent.com/AlexPnt/Default-Credit-Card-Prediction/master/figures/gui/feature-inspection-pairwise.png)

![eg](https://raw.githubusercontent.com/AlexPnt/Default-Credit-Card-Prediction/master/figures/gui/feature-inspection-edf.png)

![eg](https://raw.githubusercontent.com/AlexPnt/Default-Credit-Card-Prediction/master/figures/gui/feature-inspection-pearson.png)

![eg](https://raw.githubusercontent.com/AlexPnt/Default-Credit-Card-Prediction/master/figures/gui/feature-inspection-pca.png)

![eg](https://raw.githubusercontent.com/AlexPnt/Default-Credit-Card-Prediction/master/figures/gui/feature-inspection-lda.png)

![eg](https://raw.githubusercontent.com/AlexPnt/Default-Credit-Card-Prediction/master/figures/gui/preprocessing.png)

![eg](https://raw.githubusercontent.com/AlexPnt/Default-Credit-Card-Prediction/master/figures/gui/feature_selection.png)

![eg](https://raw.githubusercontent.com/AlexPnt/Default-Credit-Card-Prediction/master/figures/gui/metrics.png)

![eg](https://raw.githubusercontent.com/AlexPnt/Default-Credit-Card-Prediction/master/figures/gui/roc.png)

![eg](https://raw.githubusercontent.com/AlexPnt/Default-Credit-Card-Prediction/master/figures/gui/precision-recall.png)
