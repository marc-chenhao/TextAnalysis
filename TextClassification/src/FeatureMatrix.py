'''
Created on 24 Oct 2015

@author: 453334
'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import hstack
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

'''Merge two feature matrix'''
def Merge_Matrix(matrixA,matrixB):
    final_feature_matrix = hstack([matrixA,matrixB])
    return final_feature_matrix

'''bag of word, return training data feature matrix and test data feature matrix'''
def BoW_TDM(train_data,test_data):
    cont_vect = CountVectorizer(max_df=0.8, min_df=0.2,analyzer='word')
#     cont_vect = CountVectorizer(analyzer='char',ngram_range=(1,1))
    train_data_matrix = cont_vect.fit_transform(train_data)
    test_data_matrix = cont_vect.transform(test_data)
    return train_data_matrix,test_data_matrix

def Ngrams_TDM(train_data,test_data):
    cont_vect = CountVectorizer(max_df=0.8, min_df=0.2,analyzer='char',ngram_range=(2,4))
    train_data_matrix = cont_vect.fit_transform(train_data)
    test_data_matrix = cont_vect.transform(test_data)
    return train_data_matrix,test_data_matrix

'''bag of word and normalized feature to [0-1], return training data feature matrix and test data feature matrix'''
def Standard_Scaler(train_data_matrix,test_data_matrix):
    stand_scaler = StandardScaler(with_mean=False, with_std=True, copy=True)
    train_data_std_matrix = stand_scaler.fit_transform(train_data_matrix.toarray().astype(float))
    test_data_std_matrix = stand_scaler.transform(test_data_matrix.toarray().astype(float))
    return train_data_std_matrix, test_data_std_matrix

'''Feature selection chi, return training and test data feature matrix'''
def Feature_Red_CHI(train_data_matrix,test_data_matrix,train_label):
    ch2 = SelectKBest(chi2,k=200)
    train_data_matrix_FR = ch2.fit_transform(train_data_matrix,train_label)
    test_data_matrix_FR = ch2.transform(test_data_matrix)
    return train_data_matrix_FR, test_data_matrix_FR

'''Tf Feature Matrix'''
def TF_Feature_Matrix(train_data_matrix,test_data_matrix):
    transformer = TfidfTransformer(use_idf=False)
    train_data_matrix = transformer.fit_transform(train_data_matrix)
    test_data_matrix = transformer.transform(test_data_matrix)
    return train_data_matrix,test_data_matrix
    