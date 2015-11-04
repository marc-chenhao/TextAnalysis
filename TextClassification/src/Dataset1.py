'''
Created on 24 Oct 2015

@author: 453334
'''
import csv
import numpy as np
from Classifier import Classifier

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
 
def Count_Pos_Neg(inputfilename):
    All_content = []
    Label = []
    positive_content = []
    negative_content = []
    with open(inputfilename,'r',newline='') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if row['lang']=='en':
                if row['Bullying_Traces'] == 'y':
                    positive_content.append(row['text'])
                    All_content.append(row['text'])
                    Label.append(1)
                else:
                    negative_content.append(row['text'])
                    All_content.append(row['text'])
                    Label.append(0)
    return np.array(All_content),np.array(Label),np.array(positive_content),np.array(negative_content)
 
if __name__ == '__main__':
    all_content = []
    label = []
    pos_content = []
    neg_content = []
    all_content,label,pos_content,neg_content = Count_Pos_Neg('Dataset1.csv')
    Classifier(all_content,label)
#     print ('----------All data-----------')
#     Preprocessing.Count_Punc_Words(all_content)
#     Preprocessing.Count_Upper(all_content)
#     print ('----------Pos data-----------')
#     Preprocessing.Count_Punc_Words(pos_content)
#     Preprocessing.Count_Upper(pos_content)
#     print ('----------Neg data-----------')
#     Preprocessing.Count_Punc_Words(neg_content)
#     Preprocessing.Count_Upper(neg_content)
