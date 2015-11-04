'''
Created on 2 Nov 2015

@author: 453334
'''
import csv
import Preprocessing
from Classifier import Classifier
import numpy as np
import FeatureMatrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn import cross_validation

def Count_Pos_Neg(inputfilename):
    All_content = []
    Label = []
    positive_content = []
    negative_content = []
    with open(inputfilename,'r',newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if row['Label'] == 'Y':
                positive_content.append(row['Text'])
                All_content.append(row['Text'])
                Label.append(1)
            else:
                negative_content.append(row['Text'])
                All_content.append(row['Text'])
                Label.append(0)
    return np.array(All_content),np.array(Label),np.array(positive_content),np.array(negative_content)

if __name__ == '__main__':
#     all_content = []
#     label = []
#     pos_content = []
#     neg_content = []
    all_content,label,pos_content,neg_content = Count_Pos_Neg('Dataset4.csv')
#     cont_vect = CountVectorizer()
#     train_data_matrix = cont_vect.fit_transform(all_content)
#     print (train_data_matrix.shape)
#     clf = SVC(C=1.0,kernel='linear')
#     predict = clf.fit(train_data_matrix,label)
    skf = cross_validation.StratifiedKFold(label, n_folds=10, shuffle=True, random_state=None)
    '''For each fold, Do the classification'''
    mylist = list(skf)
#     train,test = mylist[0]
#     print (train)
#     print (test)
    for i in range(0,10):
        train,test = mylist[i]
        print (train)
        print (test)
        train_data = np.array(all_content[train])
        train_label = np.array(label[train])
        test_data = np.array(all_content[test])
        test_label = np.array(label[test])
        print (train_data)
        print (train_label)
        print (test_data)
        print (test_label)
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        print ('----------------------------')
    
#     for train_index, test_index in skf:
#         train_data = np.array(all_content[train_index])
#         print (train_data)
#         train_label = np.array(label[train_index])
#         test_data = np.array(all_content[test_index])
#         test_label = np.array(label[test_index])
    print ('Done')

