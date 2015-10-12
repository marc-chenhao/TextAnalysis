'''
Created on 9 Sep 2015

@author: 453334
'''
import csv
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.feature_extraction.text import CountVectorizer

"""================Data Retrieve Methods==================="""
def textRt(inputfilename):
    textContent = []
    textLabel = []
    with open(inputfilename,'r',newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if row['lang']=='en':
                textContent.append(row['text'])
                if row['Bullying_Traces'] == 'y':
                    textLabel.append(1)  
                else:
                    textLabel.append(0)
        print (textLabel.count(1))
        print (textLabel.count(0))
        return (np.array(textContent),np.array(textLabel))

def textRt_balance(inputfilename):
    textContent = []
    textLabel = []
    max_num_content = 0
    max_num_label = 0
    with open(inputfilename,'r',newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if row['lang']=='en':
                if (row['Bullying_Traces'] == 'y' and max_num_content < 1200):
                    textLabel.append(1)
                    textContent.append(row['text'])
                    max_num_content += 1
                elif (row['Bullying_Traces'] == 'n' and max_num_label < 1200):
                    textLabel.append(0)
                    textContent.append(row['text'])
                    max_num_label += 1
                else:
                    continue                        
        return (np.array(textContent),np.array(textLabel))

def textRt_Role(inputfilename):
    textContent = []
    textLabel = []
    with open(inputfilename,'r',newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if row['lang'] == 'en' and row['Bullying_Traces'] == 'y':
                textContent.append(row['text'])
                if row['Author_Role'] == 'accuser':
                    textLabel.append(1)
                elif row['Author_Role'] == 'bully':
                    textLabel.append(2)
                elif row['Author_Role'] == 'reporter':
                    textLabel.append(3)
                elif row['Author_Role'] == 'victim':
                    textLabel.append(4)
                else:
                    textLabel.append(0)
        return (np.array(textContent),np.array(textLabel))

"""=================Preprocessing======================"""    
import re

def low_case(text_data):
    text_data = text_data.lower()
    return (text_data)

def replaceURL(text_data):
    rex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    string_rp = re.sub(rex,'URLLink',text_data)
    return (string_rp)

def replaceAT(text_data):
    rex = "@([a-z0-9_]+)"
    string_rp = re.sub(rex,'@USERNAME',text_data)
    return (string_rp)

"""==========================Classification Alogrithms=================================="""
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB

def SVM_classifier(textcontent,textlabel):
    clf = SVC(C=1.0, kernel='linear')
    clf.fit(textcontent, textlabel)
    return clf

def NB_classifier(textcontent,textlabel):
    clf = BernoulliNB()
    clf.fit(textcontent, textlabel)
    return clf

from sklearn.grid_search import GridSearchCV
def parameter_setting(textContent,textLabel):
    tuned_parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    scores = ['precision','recall']
    for score in scores:
        clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring='%s_weighted' % score)
        clf.fit(textContent, textLabel)
        print("Best parameters set found on development set:")
        print(clf.best_score_)
        print(clf.best_params_)   
    return

"""========================Feature Selection=============================="""
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


"""==========================compare two algorithms=========================="""
def compare_result(label,svm_pre,nb_pre):
    both_correct = 0
    svm_correct = 0
    nb_correct = 0
    both_incorrect = 0
    for i in range(len(label)):
        if label[i] == svm_pre[i] and label[i] == nb_pre[i]:
            both_correct += 1
        elif label[i] != svm_pre[i] and label[i] != nb_pre[i]:
            both_incorrect += 1
        elif label[i] == svm_pre[i] and label[i] != nb_pre[i]:
            svm_correct += 1
        elif label[i] != svm_pre[i] and label[i] == nb_pre[i]:
            nb_correct += 1    
    return (both_correct,svm_correct,nb_correct,both_incorrect)

"""===========================count Y and N============================="""
def count_result(label,alg_pre):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(label)):
        if label[i] == 1 and alg_pre[i] == 1:
            tp += 1
        elif label[i] == 1 and alg_pre[i] == 0:
            fn += 1
        elif label[i] == 0 and alg_pre[i] == 1:
            fp += 1
        elif label[i] == 0 and alg_pre[i] == 0:
            tn += 0
    return (tp,fp,tn,fn)

"""=====================Code Begin=============================="""
textContent_raw,textLabel = textRt_balance('output.csv')

textContent_pre = []
for text in textContent_raw:
    text = low_case(text)
    text = replaceAT(text)
    text = replaceURL(text)
    textContent_pre.append(text)
textContent_raw = np.array(textContent_pre)

skf = cross_validation.StratifiedKFold(textLabel, n_folds=30, shuffle=True, random_state=None)

sum_score_svm = 0
sum_score_nb = 0
sum_recall_svm = 0
sum_recall_nb = 0

sum_score_svm_fs = 0
sum_score_nb_fs = 0
sum_recall_svm_fs = 0
sum_recall_nb_fs = 0

sum_both_correct = 0
sum_svm_correct = 0
sum_nb_correct = 0
sum_both_incorrect = 0

sum_both_correct_fs = 0
sum_svm_correct_fs = 0
sum_nb_correct_fs = 0
sum_both_incorrect_fs = 0

fold_times = 1

for train_index, test_index in skf:
#     print ("Train data index:",train_index)
#     print ("Test data index:",test_index)
    print ("This is fold %d" %fold_times)
    fold_times += 1
    X_train, X_test, y_train, y_test = textContent_raw[train_index], textContent_raw[test_index], textLabel[train_index], textLabel[test_index]
    cont_vect = CountVectorizer(ngram_range=(1, 1))
    train_data_matrix = cont_vect.fit_transform(X_train)
    
#     tuned_parameters = [{'kernel': ['linear'], 'C': [10, 1.1, 1.2, 1.3]}]
#     scores = ['accuracy']
#     for score in scores:
#         clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5)
#         clf.fit(train_data_matrix, y_train)
#         print("Best parameters set found on development set:")
#         print(clf.best_params_)
    
    """no feature selection"""
    svm_classifier = SVM_classifier(train_data_matrix, y_train)
    nb_classifier = NB_classifier(train_data_matrix, y_train)
     
    test_data_matrix = cont_vect.transform(X_test)

#     y_predict = svm_classifier.predict(test_data_matrix)
    
#     nb_classifier = NB_classifier(train_data_matrix, y_train)
#     test_data_matrix = cont_vect.transform(X_test)
# #    y_predict = nb_classifier.predict(test_data_matrix)
    """feature selection CHI"""
    ch2 = SelectKBest(chi2,k=2000)
    train_data_matrix_feature_selection = ch2.fit_transform(train_data_matrix,y_train)
    svm_classifier_feature_selection = SVM_classifier(train_data_matrix_feature_selection, y_train)
    nb_classifier_feature_selection = NB_classifier(train_data_matrix_feature_selection, y_train)
    test_data_matrix_feature_selection = cont_vect.transform(X_test)
    test_data_matrix_feature_selection = ch2.transform(test_data_matrix)
    
    
    """=============Results================="""
    y_predict_svm = svm_classifier.predict(test_data_matrix)
    y_predict_nb = nb_classifier.predict(test_data_matrix)
    
    y_predict_svm_fs = svm_classifier_feature_selection.predict(test_data_matrix_feature_selection)
    y_predict_nb_fs = nb_classifier_feature_selection.predict(test_data_matrix_feature_selection)
    
    both_correct = 0
    svm_correct = 0
    nb_correct = 0
    both_incorrect = 0
    
    """no feature selection"""
    both_correct,svm_correct,nb_correct,both_incorrect = compare_result(y_test.tolist(),y_predict_svm.tolist(),y_predict_nb.tolist())
#     print ('both_correct: ')
#     print (both_correct)
#     print ('svm_correct: ')
#     print(svm_correct)
#     print ('nb_correct: ')
#     print(nb_correct)
#     print ('both_incorrect: ')
#     print(both_incorrect)
    
    sum_both_correct += both_correct
    sum_svm_correct += svm_correct
    sum_nb_correct += nb_correct
    sum_both_incorrect += both_incorrect
    
    """feature selection"""
    both_correct,svm_correct,nb_correct,both_incorrect = compare_result(y_test.tolist(),y_predict_svm_fs.tolist(),y_predict_nb_fs.tolist())
#     print ('both_correct_fs: ')
#     print (both_correct)
#     print ('svm_correct_fs: ')
#     print(svm_correct)
#     print ('nb_correct_fs: ')
#     print(nb_correct)
#     print ('both_incorrect_fs: ')
#     print(both_incorrect)
    sum_both_correct_fs += both_correct
    sum_svm_correct_fs += svm_correct
    sum_nb_correct_fs += nb_correct
    sum_both_incorrect_fs += both_incorrect
#     tp = 0
#     fp = 0
#     tn = 0
#     fn = 0
#     tp,fp,tn,fn = count_result(y_test, y_predict_svm)
#     print (tp,fp,tn,fn)
       
    ac_svm = accuracy_score(y_test, y_predict_svm)
    ac_nb = accuracy_score(y_test, y_predict_nb)
    recall_svm = recall_score(y_test, y_predict_svm)
    recall_nb = recall_score(y_test, y_predict_nb)

    ac_svm_fs = accuracy_score(y_test, y_predict_svm_fs)
    ac_nb_fs = accuracy_score(y_test, y_predict_nb_fs)
    recall_svm_fs = recall_score(y_test, y_predict_svm_fs)
    recall_nb_fs = recall_score(y_test, y_predict_nb_fs)
    
    #f1 = f1_score(y_test, y_predict)
    print ("The accuracy of SVM for each fold: ")
    print (ac_svm)
    print ("The accuracy of N.B for each fold: ")
    print (ac_nb)
    print ("The recall of bullying using SVM for each fold: ")
    print (recall_svm)
    print ("The recall of bullying using N.B for each fold: ")
    print (recall_nb)
    print ("----------------------After Feature Selection-----------------------")
    print ("The accuracy of SVM for each fold FS: ")
    print (ac_svm_fs)
    print ("The accuracy of N.B for each fold FS: ")
    print (ac_nb_fs)
    print ("The recall of bullying using SVM for each fold FS: ")
    print (recall_svm_fs)
    print ("The recall of bullying using N.B for each fold FS: ")
    print (recall_nb_fs)

    
    sum_score_svm = sum_score_svm + ac_svm
    sum_score_nb = sum_score_nb + ac_nb
    
    sum_score_svm_fs = sum_score_svm_fs + ac_svm_fs
    sum_score_nb_fs = sum_score_nb_fs + ac_nb_fs
    
    sum_recall_svm = sum_recall_svm + recall_svm
    sum_recall_nb = sum_recall_nb + recall_nb
    
    sum_recall_svm_fs = sum_recall_svm_fs + recall_svm_fs
    sum_recall_nb_fs = sum_recall_nb_fs + recall_nb_fs
    
    print ("======================================================")

print ('The total average accuracy of SVM is:')
print (sum_score_svm/30)

print ('The total average accuracy of NB is:')
print (sum_score_nb/30)

print ('The recall of bullying using SVM:')
print (sum_recall_svm/30)

print ('The recall of bullying using N.B:')
print (sum_recall_nb/30)

print ('The total both_correct is:')
print (sum_both_correct)
print ('The total svm_correct is:')
print (sum_svm_correct)
print ('The total nb_correct is:')
print (sum_nb_correct)
print ('The total both_incorrect is:')
print (sum_both_incorrect)
print ("=======================Feature Selection===============================")
print ('The total average accuracy of SVM is:')
print (sum_score_svm_fs/30)

print ('The total average accuracy of NB is:')
print (sum_score_nb_fs/30)

print ('The recall of bullying using SVM:')
print (sum_recall_svm_fs/30)

print ('The recall of bullying using N.B:')
print (sum_recall_nb_fs/30)

print ('The total both_correct is:')
print (sum_both_correct_fs)
print ('The total svm_correct is:')
print (sum_svm_correct_fs)
print ('The total nb_correct is:')
print (sum_nb_correct_fs)
print ('The total both_incorrect is:')
print (sum_both_incorrect_fs)