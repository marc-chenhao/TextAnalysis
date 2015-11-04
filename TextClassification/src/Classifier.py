'''
Created on 24 Oct 2015

@author: 453334
'''

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import Preprocessing
import FeatureMatrix
import ShowResults
import numpy as np

'''Define classifier'''
def SVM_classifier(data,label):
    clf = SVC(C=1.0,kernel='linear')
    clf.fit(data,label)
    return clf
def NB_classifier(data,label):
    clf = MultinomialNB()
    clf.fit(data, label)
    return clf
def LR_classifier(data,label):
    clf = LogisticRegression()
    clf.fit(data, label)
    return clf
def Ensemble_classifier(class1,class2,class3):
    newlist = []
    for a,b,c in zip(class1.tolist(),class2.tolist(),class3.tolist()):
        if (a==b and a==c):
            newlist.append(a)
        elif (a==b and a!=c):
            newlist.append(a)
        elif (a!=b and a==c):
            newlist.append(a)
        elif (a!=b and a!=c):
            newlist.append(b)
    return np.array(newlist)

def Classifier(data_content,data_label):
    '''Define two global variable to save recall score'''
    positive_svm_recall = 0.0
    negative_svm_recall = 0.0
    positive_nb_recall = 0.0
    negative_nb_recall = 0.0
    positive_lr_recall = 0.0
    negative_lr_recall = 0.0
    positive_en_recall = 0.0
    negative_en_recall = 0.0
    
    positive_svm_ngrams_recall = 0.0
    negative_svm_ngrams_recall = 0.0
    positive_nb_ngrams_recall = 0.0
    negative_nb_ngrams_recall = 0.0
    positive_lr_ngrams_recall = 0.0
    negative_lr_ngrams_recall = 0.0
    positive_en_ngrams_recall = 0.0
    negative_en_ngrams_recall = 0.0
    
    '''Data Preprocessing'''
    data_content = Preprocessing.combine_all(data_content)
    '''Cross-Validation'''
    skf = cross_validation.StratifiedKFold(data_label, n_folds=10, shuffle=True, random_state=None)
    '''For each fold, Do the classification'''
    for train_index, test_index in skf:
        train_data = np.array(data_content[train_index])
        train_label = np.array(data_label[train_index])
        test_data = np.array(data_content[test_index])
        test_label = np.array(data_label[test_index])
        '''Create feature matrix'''
        train_feature_matrix, test_feature_matrix = FeatureMatrix.BoW_TDM(train_data, test_data)
        train_feature_matrix_ngrams, test_feature_matrix_ngrams = FeatureMatrix.Ngrams_TDM(train_data, test_data)
        '''TF feature matrix'''
        train_feature_matrix, test_feature_matrix = FeatureMatrix.TF_Feature_Matrix(train_feature_matrix, test_feature_matrix)
        train_feature_matrix_ngrams, test_feature_matrix_ngrams = FeatureMatrix.TF_Feature_Matrix(train_feature_matrix_ngrams, test_feature_matrix_ngrams)
        '''Merge Matrix if needs'''
        
        '''Feature Selection'''
#         train_feature_matrix, test_feature_matrix = FeatureMatrix.Feature_Red_CHI(train_feature_matrix, test_feature_matrix, train_label)
        '''Classify'''
        '''---for bag of words---'''
        svm_classifier = SVM_classifier(train_feature_matrix, train_label)
        nb_classifier = NB_classifier(train_feature_matrix, train_label)
        lr_classifier = LR_classifier(train_feature_matrix, train_label)
        predict_svm_result = svm_classifier.predict(test_feature_matrix)
        predict_nb_result = nb_classifier.predict(test_feature_matrix)
        predict_lr_result = lr_classifier.predict(test_feature_matrix)
        predict_en_result = np.array(Ensemble_classifier(predict_svm_result, predict_nb_result, predict_lr_result))
        '''---for n-grams---'''
        svm_classifier_ngrams = SVM_classifier(train_feature_matrix_ngrams, train_label)
        nb_classifier_ngrams = NB_classifier(train_feature_matrix_ngrams, train_label)
        lr_classifier_ngrams = LR_classifier(train_feature_matrix_ngrams, train_label)
        predict_svm_ngrams_result = svm_classifier_ngrams.predict(test_feature_matrix_ngrams)
        predict_nb_ngrams_result = nb_classifier_ngrams.predict(test_feature_matrix_ngrams)
        predict_lr_ngrams_result = lr_classifier_ngrams.predict(test_feature_matrix_ngrams)
        predict_en_ngrams_result = np.array(Ensemble_classifier(predict_svm_ngrams_result, predict_nb_ngrams_result, predict_lr_ngrams_result))
        '''Predict Result'''
        '''SVM'''
        recall_pos, recall_neg = ShowResults.result_recall(test_label, predict_svm_result)
        positive_svm_recall += recall_pos
        negative_svm_recall += recall_neg
        
        recall_pos, recall_neg = ShowResults.result_recall(test_label, predict_svm_ngrams_result)
        positive_svm_ngrams_recall += recall_pos
        negative_svm_ngrams_recall += recall_neg
        
        '''NB'''
        recall_pos, recall_neg = ShowResults.result_recall(test_label, predict_nb_result)
        positive_nb_recall += recall_pos
        negative_nb_recall += recall_neg
        
        recall_pos, recall_neg = ShowResults.result_recall(test_label, predict_nb_ngrams_result)
        positive_nb_ngrams_recall += recall_pos
        negative_nb_ngrams_recall += recall_neg
        
        '''LR'''
        recall_pos, recall_neg = ShowResults.result_recall(test_label, predict_lr_result)
        positive_lr_recall += recall_pos
        negative_lr_recall += recall_neg
        
        recall_pos, recall_neg = ShowResults.result_recall(test_label, predict_lr_ngrams_result)
        positive_lr_ngrams_recall += recall_pos
        negative_lr_ngrams_recall += recall_neg
        
        '''En'''
        recall_pos, recall_neg = ShowResults.result_recall(test_label, predict_en_result)
        positive_en_recall += recall_pos
        negative_en_recall += recall_neg
        
        recall_pos, recall_neg = ShowResults.result_recall(test_label, predict_en_ngrams_result)
        positive_en_ngrams_recall += recall_pos
        negative_en_ngrams_recall += recall_neg
        '''Clear the narray'''
        train_data = []
        train_label = []
        test_data = []
        test_label = []
    '''print SVM'''   
    print ('The SVM positive class recall is: %.2f' %(positive_svm_recall/10))
    print ('The SVM negative class recall is: %.2f' %(negative_svm_recall/10))
    print ('The SVM average recall is: %.2f' %((negative_svm_recall+positive_svm_recall)/20))
    print ('-----------------------------------------')
    '''print NB'''   
    print ('The NB positive class recall is: %.2f' %(positive_nb_recall/10))
    print ('The NB negative class recall is: %.2f' %(negative_nb_recall/10))
    print ('The NB average recall is: %.2f' %((negative_nb_recall+positive_nb_recall)/20))
    print ('-----------------------------------------')
    '''print LR'''   
    print ('The LR positive class recall is: %.2f' %(positive_lr_recall/10))
    print ('The LR negative class recall is: %.2f' %(negative_lr_recall/10))
    print ('The LR average recall is: %.2f' %((negative_lr_recall+positive_lr_recall)/20))
    print ('-----------------------------------------')
    '''print EN'''   
    print ('The EN positive class recall is: %.2f' %(positive_en_recall/10))
    print ('The EN negative class recall is: %.2f' %(negative_en_recall/10))
    print ('The EN average recall is: %.2f' %((negative_en_recall+positive_en_recall)/20))
    
    print ('*****************************************')
    '''print SVM'''
    print ('The SVM positive class recall is: %.2f' %(positive_svm_ngrams_recall/10))
    print ('The SVM negative class recall is: %.2f' %(negative_svm_ngrams_recall/10))
    print ('The SVM average recall is: %.2f' %((negative_svm_ngrams_recall+positive_svm_ngrams_recall)/20))
    print ('-----------------------------------------')
    '''print NB'''   
    print ('The NB positive class recall is: %.2f' %(positive_nb_ngrams_recall/10))
    print ('The NB negative class recall is: %.2f' %(negative_nb_ngrams_recall/10))
    print ('The NB average recall is: %.2f' %((negative_nb_ngrams_recall+positive_nb_ngrams_recall)/20))
    print ('-----------------------------------------')
    '''print LR'''   
    print ('The LR positive class recall is: %.2f' %(positive_lr_ngrams_recall/10))
    print ('The LR negative class recall is: %.2f' %(negative_lr_ngrams_recall/10))
    print ('The LR average recall is: %.2f' %((negative_lr_ngrams_recall+positive_lr_ngrams_recall)/20))
    print ('-----------------------------------------')
    '''print EN'''   
    print ('The EN positive class recall is: %.2f' %(positive_en_ngrams_recall/10))
    print ('The EN negative class recall is: %.2f' %(negative_en_ngrams_recall/10))
    print ('The EN average recall is: %.2f' %((negative_en_ngrams_recall+positive_en_ngrams_recall)/20))
    return


# '''Chose data set'''
# data_content,data_label = Dataset1.textRt('output.csv')
# # data_content, data_label = Dataset1.textRt_balance('output.csv')
# 
# '''Data Preprocessing'''
# data_content = Preprocessing.combine_all(data_content)
# 
# '''Cross-Validation'''
# skf = cross_validation.StratifiedKFold(data_label, n_folds=10, shuffle=True, random_state=None)
# 
# '''For each fold, Do the classification'''
# for train_index, test_index in skf:
#     train_data = data_content[train_index]
#     train_label = data_label[train_index]
#     test_data = data_content[test_index]
#     test_label = data_label[test_index]
#     '''Create feature matirx'''
#     train_feature_matrix, test_feature_matrix = FeatureMatrix.BoW_TDM(train_data, test_data)
#     '''Normalized feature matrix'''
#     train_feature_matrix, test_feature_matrix = FeatureMatrix.Standard_Scaler(train_feature_matrix, test_feature_matrix)
#     '''Merge Matrix if needs'''
#     
#     '''Feature Selection'''
#     train_feature_matrix, test_feature_matrix = FeatureMatrix.Feature_Red_CHI(train_feature_matrix, test_feature_matrix, train_label)
#     '''Classify'''
#     svm_classifier = SVM_classifier(train_feature_matrix, train_label)
#     predict_result = svm_classifier.predict(test_feature_matrix)
#     '''Result'''
#     recall_pos, recall_neg = ShowResults.result_recall(test_label, predict_result)
#     positive_recall += recall_pos
#     negative_recall += recall_neg
# print ('The positive class recall is: %.2f' %(positive_recall/10))
# print ('The negative class recall is: %.2f' %(negative_recall/10))
