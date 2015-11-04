'''
Created on 29 Oct 2015

@author: 453334
'''
import csv
import Preprocessing
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
    Classifier(all_content[:10000],label[:10000])