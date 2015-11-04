'''
Created on 28 Oct 2015

@author: 453334
'''
import csv
from Classifier import Classifier
import numpy as np

def Count_Pos_Neg(inputfilename):
    All_content = []
    Label = []
    positive_content = []
    negative_content = []
    with open(inputfilename,'r',newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if row['Class'] == '1':
                positive_content.append(row['Comments'])
                All_content.append(row['Comments'])
                Label.append(1)
            else:
                negative_content.append(row['Comments'])
                All_content.append(row['Comments'])
                Label.append(0)
    return All_content,Label,positive_content,negative_content

if __name__ == '__main__':
#     all_content = []
#     label = []
#     pos_content = []
#     neg_content = []
    all_content,label,pos_content,neg_content = Count_Pos_Neg('Dataset2.csv')
#     print ('The total number of text content is: %.2f' %len(all_content))
#     print ('The total number of positive content is: %.2f and the proportion is %.4f' %(len(pos_content),len(pos_content)/len(all_content)))
#     print ('The total number of negative content is: %.2f and the proportion is %.4f' %(len(neg_content),len(neg_content)/len(all_content)))
#     print ('----------All data-----------')
#     Preprocessing.Count_Punc_Words(all_content)
#     Preprocessing.Count_Upper(all_content)
#     print ('----------Pos data-----------')
#     Preprocessing.Count_Punc_Words(pos_content)
#     Preprocessing.Count_Upper(pos_content)
#     print ('----------Neg data-----------')
#     Preprocessing.Count_Punc_Words(neg_content)
#     Preprocessing.Count_Upper(neg_content)
#     all_content,label,pos_content,neg_content = Count_Pos_Neg('Dataset2.csv')
#     all_content = pos_content[:100] + neg_content[:100]
#     labely = [1] * 100
#     labeln = [0] * 100
#     label = labely + labeln
    Classifier(np.array(all_content),np.array(label))