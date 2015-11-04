'''
Created on 30 Oct 2015

@author: 453334
'''
import csv
import Preprocessing
import numpy as np
from Classifier import Classifier

def Count_Pos_Neg(inputfilename):
    All_content = []
    Label = []
    positive_content = []
    negative_content = []
    with open(inputfilename,'r',newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if row['Label'] == '1':
                positive_content.append(row['post_body:'])
                All_content.append(row['post_body:'])
                Label.append(1)
            else:
                negative_content.append(row['post_body:'])
                All_content.append(row['post_body:'])
                Label.append(0)
    return All_content,Label,positive_content,negative_content

if __name__ == '__main__':
    all_content = []
    label = []
    pos_content = []
    neg_content = []
    all_content,label,pos_content,neg_content = Count_Pos_Neg('Dataset6.csv')
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
#     all_content = pos_content[:50] + neg_content[:50]
#     labely = [1] * 50
#     labeln = [0] * 50
#     label = labely + labeln
    Classifier(np.array(all_content),np.array(label))