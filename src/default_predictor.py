#usr/bin/env python2
# -*- coding: utf-8 -*

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from ui import design
from core import loading,visualization,evaluation,classification
from core.feature_selection import chi2_feature_test,kw_feature_test,information_gain,gain_ratio,pearson_correlation_matrix
from core.feature_selection import information_gain_selection,gain_ratio_selection
from core.feature_selection import chi_squared_selection,kruskal_wallis_selection
from core.feature_selection import fisher_score_selection,mrmr_selection,auc_selection,rfe_selection,sfs_selection
from core.feature_selection import pearson_between_feature,pearson_between_feature_class
from core.feature_selection import pca_selection,lda_selection
from core.preprocessing import standardize_features,scale_by_min_max,scale_by_max_value,normalize
from core.preprocessing import random_undersampling,nearmiss_undersampling,ncl_undersampling,random_oversampling,smote_oversampling
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class DefaultPredictorGUI(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()

        #variables
        self.dataset_loaded=False
        self.backup_X=None
        self.backup_y=None
        self.backup_features=None
        self.backup_feature_names=None

        self.X=None
        self.y=None
        self.features=None
        self.feature_names=None

        #setup user interface
        self.setupUi(self)
        self.feature2_input.hide()
        self.kind_of_diagonal_plot.hide()
        self.diagonal_subplots_label.hide()
        self.correlation_table.hide()
        self.threshold_label.hide()
        self.threshold_doubleSpinBox.hide()
        self.sel_method_label.hide()
        self.sel_method_comboBox.hide()

        self.metric_distance_label.hide()
        self.metric_dist_comboBox.hide()
        self.k_neighbors_label.hide()
        self.k_neighbors_spinBox.hide()
        self.kernel_function_radio.hide()
        self.kernenl_function_comboBox.hide()
        self.reg_constant_label.hide()
        self.c_doubleSpinBox.hide()
        self.gamma_label.hide()
        self.gamma_doubleSpinBox.hide()
        self.n_trees_label.hide()
        self.n_trees_spinBox.hide()

        #setup callbacks
        self.browse_file_button.clicked.connect(self.browse_file)
        self.select_feature_ok.clicked.connect(self.get_feature_stats)

        self.histogram_dist.toggled.connect(self.histogram_dist_clicked)
        self.pairwise_histogram.toggled.connect(self.pairwise_histogram_clicked)
        self.box_plot.toggled.connect(self.boxplot_clicked)
        self.cdf.toggled.connect(self.cdf_clicked)
        self.pca_2d.toggled.connect(self.pca_2d_clicked)
        self.lda.toggled.connect(self.lda_clicked)
        self.plot_figure_button.clicked.connect(self.plot_feature_inspection_figure)
        self.pearson_correlation.toggled.connect(self.show_pearson_correlation)

        self.transform_button.clicked.connect(self.tranform_data)
        self.resample_button.clicked.connect(self.balance_dataset)

        self.reset_button.clicked.connect(self.reset_original_data)
        self.reset_button2.clicked.connect(self.reset_original_data)
        self.reset_button3.clicked.connect(self.reset_original_data)
        self.reset_button4.clicked.connect(self.reset_original_data)


        self.information_gain_radio.toggled.connect(self.hide_threshold_sel_method)
        self.gain_ratio_radio.toggled.connect(self.hide_threshold_sel_method)
        self.chi_squared_radio.toggled.connect(self.hide_threshold_sel_method)
        self.kw_radio.toggled.connect(self.hide_threshold_sel_method)
        self.fisher_score_radio.toggled.connect(self.hide_threshold_sel_method)
        self.pearson_corr_fc_radio.toggled.connect(self.show_threshold)
        self.pearson_corr_ff_radio.toggled.connect(self.show_threshold)
        self.mRMR_radio.toggled.connect(self.hide_threshold_sel_method)
        self.auc_radio.toggled.connect(self.hide_threshold_sel_method)
        self.mRMR_radio.toggled.connect(self.show_mrmr_sel_method)


        self.fs_pushButton.clicked.connect(self.apply_feature_selection)
        self.fr_pushButton.clicked.connect(self.apply_feature_reduction)

        self.min_dist_clf_radio.toggled.connect(self.min_dist_clicked)
        self.knn_radio.toggled.connect(self.knn_clicked)
        self.naive_bayes_radio.toggled.connect(self.nb_clicked)
        self.svm_radio.toggled.connect(self.svm_clicked)
        self.cart_radio.toggled.connect(self.cart_clicked)
        self.random_forest_radio.toggled.connect(self.random_forest_clicked)
        self.train_test_pushButton.clicked.connect(self.train_test_evaluate)



    def browse_file(self):
        try:

            #read file path
            self.file_info=QtWidgets.QFileDialog.getOpenFileName(self,"Choose Training Data")
            self.path_text.setText(self.file_info[0])

            #read training data
            self.features,self.feature_names,self.X,self.y=loading.load_dataset(self.file_info[0])
            # self.features,self.feature_names,self.X,self.y=loading.load_dataset("../dataset/default of credit card clients.csv")
            self.backup_X=self.X
            self.backup_y=self.y
            self.backup_features=self.features
            self.backup_feature_names=self.feature_names

            #setup some gui elements
            self.n_features_spinBox.setMaximum(len(self.X[0]))
            self.n_components_spinBox.setMaximum(len(self.X[0]))

            #read dataset characteristics
            self.dataset_description=self.get_dataset_description()

            #write to GUI
            self.dataset_text.setText(self.dataset_description)
            self.dataset_loaded=True
        except:
            QtWidgets.QMessageBox.information(self, "Invalid CSV file","Please enter a valid CSV file.")


    def get_dataset_description(self):
        description= """Number of Samples:"""+str(len(self.X))\
                                     +"""\nNumber of Features (Dimensionality): """+str(len(self.X[0]))\
                                     +"""\nNumber of samples from class '0': """+str(len(self.X[self.y==0,0]))\
                                     +"""\nNumber of samples from class '1': """+str(len(self.X[self.y==1,0]))\
                                     +"""\n\nFeatures:\n"""
        #read feature names
        for i in xrange(len(self.features)):
                description+=self.features[i]+": "+self.feature_names[i]+"\n"


        #fill combo box
        self.feature_input.clear()
        self.feature1_input.clear()
        self.feature2_input.clear()
        for i in xrange(len(self.features)-1):
            self.feature_input.addItem(self.features[i])
            self.feature1_input.addItem(self.features[i])
            self.feature2_input.addItem(self.features[i])

        return description

    def get_feature_stats(self):

        # #get input feature
        feature_input=self.feature_input.currentText()
        try:
            if feature_input[0]=='X':
                try:
                    feature_index=int("".join(feature_input[1:]))
                    feature_index-=1

                except:
                    QtWidgets.QMessageBox.information(self, "Wrong Format","Please enter a feature name in the format: X%d.")
                    return
            elif "".join(feature_input[0]+feature_input[1])=='LD' or "".join(feature_input[0]+feature_input[1])=='PC':
                try:
                    feature_index=int("".join(feature_input[2:]))
                    feature_index-=1

                except:
                    QtWidgets.QMessageBox.information(self, "Wrong Format","Please enter a feature name in the format: X||LD||PC%d.")
                    return
            else:
                QtWidgets.QMessageBox.information(self, "Wrong Format","Feature names must be in the format: X%d.")
                return
        except:
            QtWidgets.QMessageBox.information(self, "Data Not Found","Please load a dataset first.")
            return


        try:
            max_value=self.X[:,feature_index].max()
            min_value=self.X[:,feature_index].min()
            mean_value=self.X[:,feature_index].mean()
            std_value=self.X[:,feature_index].std()
            var_value=self.X[:,feature_index].var()
            skewness=stats.skew(self.X[:,feature_index])
            kurtosis=stats.kurtosis(self.X[:,feature_index],fisher=True)
            chi2,chi_p_val=chi2_feature_test(self.X,self.y,int(feature_index))
            H_kw,kw_p_val=kw_feature_test(self.X,self.y,int(feature_index))
            info_gain=information_gain(self.X,self.y,int(feature_index))
            gain_rt=gain_ratio(self.X,self.y,int(feature_index))

        except:
            QtWidgets.QMessageBox.information(self, "Wrong Index","Feature Index Out Of Bounds.")
            return

        feature_stats="""Statistics:\n\nMinimum Value: """+str(min_value)\
                      +"""\n\nMaximum Value: """+str(max_value)\
                      +"""\n\nMean: """+str(mean_value)\
                      +"""\n\nStandard Deviation: """+str(std_value)\
                      +"""\n\nVariance: """+str(var_value)\
                      +"""\n\nSkewness: """+str(skewness)\
                      +"""\n\nKurtosis: """+str(kurtosis)\
                      +"""\n\nChi Squared Test: """+str(chi2[0])\
                      +"""\n\nKruskal-Wallis Test:  """+str(H_kw)\
                      +"""\n\nInformation Gain: """+str(info_gain)\
                      +"""\n\nGain Ratio: """+str(gain_rt)

        self.feature_stats.setText(feature_stats)

    def plot_feature_inspection_figure(self):
        if self.dataset_loaded:
            #get selected feature
            feature_input=self.feature1_input.currentText()

            #pot desired figure
            if self.histogram_dist.isChecked():
                visualization.visualize_feature_hist_dist(self.X,self.y,feature_input,self.features,normalize=True)
                self.show_visualization("img/"+feature_input+"_hist_dist.png")
            elif self.box_plot.isChecked():
                visualization.visualize_feature_boxplot(self.X,self.y,feature_input,self.features)
                self.show_visualization("img/"+feature_input+"_boxplot.png")
            elif self.pairwise_histogram.isChecked():
                feature2_input=self.feature2_input.currentText()
                diag_kind=self.kind_of_diagonal_plot.currentText()
                if diag_kind=="Histogram":
                    diag_kind='hist'
                else:
                    diag_kind='kde'
                visualization.visualize_hist_pairplot(self.X,self.y,feature_input,feature2_input,self.features,diag_kind)
                self.show_visualization("img/"+feature_input+"_"+feature2_input+"_hist_pairplot.png")
            elif self.cdf.isChecked():
                visualization.feature_cdf(self.X,self.y,feature_input)
                self.show_visualization("img/"+feature_input+"_cdf.png")
            elif self.pca_2d.isChecked():
                visualization.visualize_pca2D(self.X,self.y)
                self.show_visualization("img/pca2D.png")
            elif self.lda.isChecked():
                visualization.visualize_lda2D(self.X,self.y)
                self.show_visualization('img/lda.png')
        else:
            QtWidgets.QMessageBox.information(self, "Data Not Found","Please load a dataset first.")
            return

    #show/hide buttons
    def pairwise_histogram_clicked(self,enabled):
        if enabled:
            self.select_feature1_label.show()
            self.feature1_input.show()
            self.feature2_input.show()
            self.kind_of_diagonal_plot.show()
            self.diagonal_subplots_label.show()

    def histogram_dist_clicked(self,enabled):
        if enabled:
            self.select_feature1_label.show()
            self.feature1_input.show()
            self.feature2_input.hide()
            self.kind_of_diagonal_plot.hide()
            self.diagonal_subplots_label.hide()

    def boxplot_clicked(self,enabled):
        if enabled:
            self.select_feature1_label.show()
            self.feature1_input.show()
            self.feature2_input.hide()
            self.kind_of_diagonal_plot.hide()
            self.diagonal_subplots_label.hide()
    def cdf_clicked(self,enabled):
        if enabled:
            self.select_feature1_label.show()
            self.feature1_input.show()
            self.feature2_input.hide()
            self.kind_of_diagonal_plot.hide()
            self.diagonal_subplots_label.hide()
    def pca_2d_clicked(self,enabled):
        if enabled:
            self.select_feature1_label.hide()
            self.feature1_input.hide()
            self.feature2_input.hide()
            self.kind_of_diagonal_plot.hide()
            self.diagonal_subplots_label.hide()

    def lda_clicked(self,enabled):
        if enabled:
            self.select_feature1_label.hide()
            self.feature1_input.hide()
            self.feature2_input.hide()
            self.kind_of_diagonal_plot.hide()
            self.diagonal_subplots_label.hide()

    def hide_threshold_sel_method(self,enabled):
        if enabled:
            self.threshold_label.hide()
            self.threshold_doubleSpinBox.hide()
            self.sel_method_label.hide()
            self.sel_method_comboBox.hide()

    def show_threshold(self,enabled):
        if enabled:
            self.threshold_label.show()
            self.threshold_doubleSpinBox.show()
            self.sel_method_label.hide()
            self.sel_method_comboBox.hide()

    def show_mrmr_sel_method(self,enabled):
        if enabled:
            self.threshold_label.hide()
            self.threshold_doubleSpinBox.hide()
            self.sel_method_label.show()
            self.sel_method_comboBox.show()

    #show figure in the visualization tab
    def show_visualization(self,fig_name,scaled=True):
        self.correlation_table.hide()
        self.feature_inspection_fig.show()
        pixmap = QtGui.QPixmap(fig_name)
        self.feature_inspection_fig.setPixmap(pixmap)
        self.feature_inspection_fig.setScaledContents(scaled)
        self.feature_inspection_fig.show()

    #show table with correlation data
    def show_pearson_correlation(self):
        if self.dataset_loaded:
            self.feature_inspection_fig.hide()
            self.correlation_table.show()
            correlation_matrix=pearson_correlation_matrix(self.X)


            #initialize rows, columns and headers
            self.correlation_table.setRowCount(0)
            self.correlation_table.setColumnCount(0)
            for i in xrange(len(self.X[0])):
                self.correlation_table.insertRow(i)
                self.correlation_table.insertColumn(i)
                self.correlation_table.setHorizontalHeaderItem(i,QtWidgets.QTableWidgetItem(self.features[i]))
                self.correlation_table.setVerticalHeaderItem(i,QtWidgets.QTableWidgetItem(self.features[i]))

            #fill table data
            for i in xrange(len(self.X[0])):
                for j in xrange(len(self.X[0])):
                    self.correlation_table.setItem(i,j,QtWidgets.QTableWidgetItem(str(correlation_matrix[i][j])))
        else:
            QtWidgets.QMessageBox.information(self, "Data Not Found","Please load a dataset first.")

    #Apply data tranformation
    def tranform_data(self):
        if self.dataset_loaded:
            if self.standardize_radio.isChecked():
                self.X=standardize_features(self.X)
                QtWidgets.QMessageBox.information(self, "Data Transformed","Success: Data centered to mean and scaled to unit variance.")
            elif self.scale_minmax_radio.isChecked():
                self.X=scale_by_min_max(self.X)
                QtWidgets.QMessageBox.information(self, "Data Transformed","Success: Data scaled to range [0,1].")
            elif self.scale_absmax_radio.isChecked():
                self.X=scale_by_max_value(self.X)
                QtWidgets.QMessageBox.information(self, "Data Transformed","Success: Data scaled to range [-1,1].")
            elif self.normalize_samples_radio.isChecked():
                self.X=normalize(self.X,axis=1)
                QtWidgets.QMessageBox.information(self, "Data Transformed","Success: Samples scaled to unit norm.")
            elif self.normalize_features_radio.isChecked():
                self.X=normalize(self.X,axis=0)
                QtWidgets.QMessageBox.information(self, "Data Transformed","Success: Features scaled to unit norm.")
        else:
            QtWidgets.QMessageBox.information(self, "Data Not Found","Please load a dataset first.")

    #balance dataset using the selected method
    def balance_dataset(self):
        if self.dataset_loaded:
            if self.random_majority_radio.isChecked():
                self.X,self.y=random_undersampling(self.X,self.y)
            elif self.nearmiss_1_radio.isChecked():
                self.X,self.y=nearmiss_undersampling(self.X,self.y,1)
            elif self.nearmiss_3_radio.isChecked():
                self.X,self.y=nearmiss_undersampling(self.X,self.y,3)
            elif self.ncl_radio.isChecked():
                self.X,self.y=ncl_undersampling(self.X,self.y)
            elif self.random_minority_radio.isChecked():
                self.X,self.y=random_oversampling(self.X,self.y)
            elif self.smote_radio.isChecked():
                self.X,self.y=smote_oversampling(self.X,self.y)

            #update dataset characteristics
            self.dataset_description=self.get_dataset_description()
            self.dataset_text.setText(self.dataset_description)
            QtWidgets.QMessageBox.information(self, "Dataset Balanced","Success: Dataset is now balanced.")
        else:
             QtWidgets.QMessageBox.information(self, "Data Not Found","Please load a dataset first.")

    #apply the selected feature selection method
    def apply_feature_selection(self):
        if self.dataset_loaded:
            feature_selected=False
            if self.information_gain_radio.isChecked():
                self.X,feature_indexes=information_gain_selection(self.X,self.y,self.n_features_spinBox.value())
                feature_selected=True
            elif self.gain_ratio_radio.isChecked():
                self.X,feature_indexes=gain_ratio_selection(self.X,self.y,self.n_features_spinBox.value())
                feature_selected=True
            elif self.chi_squared_radio.isChecked():
                self.X,feature_indexes=chi_squared_selection(self.X,self.y,self.n_features_spinBox.value())
                feature_selected=True
            elif self.kw_radio.isChecked():
                self.X,feature_indexes=kruskal_wallis_selection(self.X,self.y,self.n_features_spinBox.value())
                feature_selected=True
            elif self.fisher_score_radio.isChecked():
                self.X,feature_indexes=fisher_score_selection(self.X,self.y,self.n_features_spinBox.value())
                feature_selected=True
            elif self.pearson_corr_ff_radio.isChecked():
                self.X,feature_indexes=pearson_between_feature(self.X,self.threshold_doubleSpinBox.value())
                if len(feature_indexes)==0:
                    feature_selected=False
                    QtWidgets.QMessageBox.information(self, "No Feature selected","Warning: All Features kept.")
                else:
                    feature_selected=True
            elif self.pearson_corr_fc_radio.isChecked():
                self.X,feature_indexes=pearson_between_feature_class(self.X,self.y,self.threshold_doubleSpinBox.value())
                if len(feature_indexes)==0:
                    feature_selected=False
                    QtWidgets.QMessageBox.information(self, "No Feature selected","Warning: All Features kept.")
                else:
                    feature_selected=True
            elif self.mRMR_radio.isChecked():
                selection_method='MID'
                if self.sel_method_comboBox.currentText()=='Multiplicative Combination':
                    selection_method='MIQ'
                self.X,feature_indexes=mrmr_selection(self.X,self.features,self.y,self.n_features_spinBox.value(),selection_method)
                feature_selected=True
            elif self.auc_radio.isChecked():
                self.X,feature_indexes=auc_selection(self.X,self.y,self.n_features_spinBox.value())
                feature_selected=True
            elif self.rfe_radio.isChecked():
                self.X,feature_indexes=rfe_selection(self.X,self.y,self.n_features_spinBox.value())
                feature_selected=True
            elif self.sfs_radio.isChecked():
                self.X,feature_indexes=sfs_selection(self.X,self.y,self.n_features_spinBox.value(),forward=True)
                feature_selected=True
            elif self.sbs_radio.isChecked():
                self.X,feature_indexes=sfs_selection(self.X,self.y,self.n_features_spinBox.value(),forward=False)
                feature_selected=True

            if feature_selected:
                #update dataset characteristics
                self.update_reduced_feature_names(len(self.X[0]),'X','',feature_indexes)
                self.dataset_description=self.get_dataset_description()
                self.dataset_text.setText(self.dataset_description)
                QtWidgets.QMessageBox.information(self, "Features Selected","Success: "+str(len(self.X[0]))+" Features kept.")
                feature_selected=False
        else:
             QtWidgets.QMessageBox.information(self, "Data Not Found","Please load a dataset first.")


    #apply the selected feature reduction method
    def apply_feature_reduction(self):
        if self.dataset_loaded:
            if self.pca_radio.isChecked():
                self.X=pca_selection(self.X,self.n_components_spinBox.value())

                #update dataset characteristics
                self.update_reduced_feature_names(self.n_components_spinBox.value(),'PC','Principal Component')
                self.dataset_description=self.get_dataset_description()
                self.dataset_text.setText(self.dataset_description)

                QtWidgets.QMessageBox.information(self, "Features Reduced","Success: "+str(self.n_components_spinBox.value())+" Principal Components kept.")
            elif self.lda_radio.isChecked():
                self.X=lda_selection(self.X,self.y,self.n_components_spinBox.value())

                #update dataset characteristics
                self.update_reduced_feature_names(self.n_components_spinBox.value(),'LD','Linear Discriminant')
                self.dataset_description=self.get_dataset_description()
                self.dataset_text.setText(self.dataset_description)

                QtWidgets.QMessageBox.information(self, "Features Reduced","Success: "+str(self.n_components_spinBox.value())+" Discriminant Directions kept.")

        else:
             QtWidgets.QMessageBox.information(self, "Data Not Found","Please load a dataset first.")

    def update_reduced_feature_names(self,n,abbr_fname,full_fname,feature_indexes=None):
        self.features=[]
        self.feature_names=[]
        for i in xrange(n):
            self.features+=[abbr_fname+str(i+1)]
            if feature_indexes is None:
                self.feature_names+=[full_fname]
            else:
                self.feature_names+=[self.backup_feature_names[feature_indexes[i]]]
        self.features+=['Y']
        self.feature_names+=['default payment next month']

    #run the selected classifier
    def train_test_evaluate(self):
        if self.dataset_loaded:
            K=self.k_folds_spinBox.value()
            clf=None
            if self.min_dist_clf_radio.isChecked():
                dist=str(self.metric_dist_comboBox.currentText().lower())
                clf=classification.MinimumDistanceClassifier(dist)
            elif self.knn_radio.isChecked():
                dist=str(self.metric_dist_comboBox.currentText().lower())
                k=self.k_neighbors_spinBox.value()
                clf=KNeighborsClassifier(n_neighbors=k,metric=dist)
            elif self.naive_bayes_radio.isChecked():
                clf=GaussianNB()
            elif self.svm_radio.isChecked():
                kernel_func=self.kernenl_function_comboBox.currentText()
                c=self.c_doubleSpinBox.value()
                gamma=self.gamma_doubleSpinBox.value()

                if kernel_func=='Linear':
                    kernel_func='linear'
                elif kernel_func=='Polymonial':
                    kernel_func='poly'
                elif kernel_func=='Radial Basis Function':
                    kernel_func='rbf'
                elif kernel_func=='Sigmoid':
                    kernel_func='sigmoid'

                clf=SVC(C=c,kernel=kernel_func,gamma=gamma)

            elif self.cart_radio.isChecked():
                clf=DecisionTreeClassifier()
            elif self.random_forest_radio.isChecked():
                n_trees=self.n_trees_spinBox.value()
                clf=RandomForestClassifier(n_estimators=n_trees)

            if clf is not None:
                self.perform_evaluation(clf,K)


    #perform evaluation metric and display them in the gui
    def perform_evaluation(self,classifier,K):
        print '\nComputing ROC Curves ...'
        evaluation.visualize_k_fold_roc_plot(self.X,self.y,classifier,K)
        print '\nComputing Precision-Recall Curves ...'
        evaluation.visualize_k_fold_precision_recall_plot(self.X,self.y,classifier,K)
        print '\nGathering Metrics ...'
        confusion_matrix,accuracy,precision,recall,f1,avg_precision_recall,auc=evaluation.get_k_fold_metrics(self.X,self.y,classifier,K)

        results='=== Stratified '+str(K)+'-fold Cross-Validation ===\n\n'\
                +'Correctly Classified Instances (Accuracy)\t'+str(round(accuracy[0],2))+'+- '+str(round(accuracy[1],2))\
                +'\nPrecision \t\t\t\t'+str(round(precision[0],2))+'+- '+str(round(precision[1],2))\
                +'\nRecall\t\t\t\t'+str(round(recall[0],2))+'+- '+str(round(recall[1],2))\
                +'\nF1 score\t\t\t\t'+str(round(f1[0],2))+'+- '+str(round(f1[1],2))\
                +'\nAverage Precision\t\t\t'+str(round(avg_precision_recall[0],2))+'+- '+str(round(avg_precision_recall[1],2))\
                +'\nArea Under the Curve (AUC)\t\t'+str(round(auc[0],2))+'+- '+str(round(auc[1],2))\
                +'\n\n=== Confusion Matrix ===\n'+'\t0 \t\t1\n'\
                +'0\t'+str(int(confusion_matrix[0][0]))+' (TN)\t\t'+str(int(confusion_matrix[0][1]))+' (FP)\n'\
                +'1\t'+str(int(confusion_matrix[1][0]))+' (FN)\t\t'+str(int(confusion_matrix[1][1]))+' (TP)'

        self.results_textBrowser.setText(results)
        self.show_curve(self.roc_curves_fig,'img/roc.png')
        self.show_curve(self.pr_curves_fig,'img/pr_curve.png')

    #show figure in the visualization tab
    def show_curve(self,fig_obj,fig_name,scaled=True):
        pixmap = QtGui.QPixmap(fig_name)
        fig_obj.setPixmap(pixmap)
        fig_obj.setScaledContents(scaled)
        fig_obj.show()

    #show/hide items
    def min_dist_clicked(self,enabled):
        if enabled:
            self.metric_distance_label.show()
            self.metric_dist_comboBox.show()

            self.k_neighbors_label.hide()
            self.k_neighbors_spinBox.hide()
            self.kernel_function_radio.hide()
            self.kernenl_function_comboBox.hide()
            self.reg_constant_label.hide()
            self.c_doubleSpinBox.hide()
            self.gamma_label.hide()
            self.gamma_doubleSpinBox.hide()
            self.n_trees_label.hide()
            self.n_trees_spinBox.hide()

    def knn_clicked(self,enabled):
        if enabled:
            self.metric_distance_label.show()
            self.metric_dist_comboBox.show()
            self.k_neighbors_label.show()
            self.k_neighbors_spinBox.show()

            self.kernel_function_radio.hide()
            self.kernenl_function_comboBox.hide()
            self.reg_constant_label.hide()
            self.c_doubleSpinBox.hide()
            self.gamma_label.hide()
            self.gamma_doubleSpinBox.hide()
            self.n_trees_label.hide()
            self.n_trees_spinBox.hide()

    def nb_clicked(self,enabled):
        if enabled:
            self.metric_distance_label.hide()
            self.metric_dist_comboBox.hide()
            self.k_neighbors_label.hide()
            self.k_neighbors_spinBox.hide()
            self.kernel_function_radio.hide()
            self.kernenl_function_comboBox.hide()
            self.reg_constant_label.hide()
            self.c_doubleSpinBox.hide()
            self.gamma_label.hide()
            self.gamma_doubleSpinBox.hide()
            self.n_trees_label.hide()
            self.n_trees_spinBox.hide()

    def svm_clicked(self,enabled):
        if enabled:
            self.kernel_function_radio.show()
            self.kernenl_function_comboBox.show()
            self.reg_constant_label.show()
            self.c_doubleSpinBox.show()
            self.gamma_label.show()
            self.gamma_doubleSpinBox.show()

            self.metric_distance_label.hide()
            self.metric_dist_comboBox.hide()
            self.k_neighbors_label.hide()
            self.k_neighbors_spinBox.hide()
            self.n_trees_label.hide()
            self.n_trees_spinBox.hide()

    def cart_clicked(self,enabled):
        if enabled:
            self.metric_distance_label.hide()
            self.metric_dist_comboBox.hide()
            self.k_neighbors_label.hide()
            self.k_neighbors_spinBox.hide()
            self.kernel_function_radio.hide()
            self.kernenl_function_comboBox.hide()
            self.reg_constant_label.hide()
            self.c_doubleSpinBox.hide()
            self.gamma_label.hide()
            self.gamma_doubleSpinBox.hide()
            self.n_trees_label.hide()
            self.n_trees_spinBox.hide()


    def random_forest_clicked(self,enabled):
        if enabled:
            self.n_trees_label.show()
            self.n_trees_spinBox.show()

            self.metric_distance_label.hide()
            self.metric_dist_comboBox.hide()
            self.k_neighbors_label.hide()
            self.k_neighbors_spinBox.hide()
            self.kernel_function_radio.hide()
            self.kernenl_function_comboBox.hide()
            self.reg_constant_label.hide()
            self.c_doubleSpinBox.hide()
            self.gamma_label.hide()
            self.gamma_doubleSpinBox.hide()


    #set original data back
    def reset_original_data(self):
        if self.dataset_loaded:
            self.X=self.backup_X
            self.y=self.backup_y
            self.features=self.backup_features
            self.feature_names=self.backup_feature_names

            #update dataset characteristics
            self.dataset_description=self.get_dataset_description()
            self.dataset_text.setText(self.dataset_description)

            QtWidgets.QMessageBox.information(self, "Reset","Success: Data replace with original data.")
        else:
            QtWidgets.QMessageBox.information(self, "Data Not Found","Please load a dataset first.")



def main():
    predictor = QtWidgets.QApplication(sys.argv)  	#instanciate
    gui = DefaultPredictorGUI()  					#create gui
    gui.show()  									#show gui
    predictor.exec_()								#run application


if __name__ == '__main__':
    main()  # run app
