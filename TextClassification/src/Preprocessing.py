'''
Created on 26 Oct 2015

@author: 453334
'''
import re
import numpy as np
from string import punctuation
from collections import Counter
import nltk

'''upper_case convert to lower_case'''
def low_case(text_data):
    text_data = text_data.lower()
    return (text_data)
'''URL replace'''
def replaceURL(text_data):
    rex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    string_rp = re.sub(rex,'URLLink',text_data)
    return (string_rp)
'''username replace'''
def replaceAT(text_data):
    rex = "@([a-z0-9_]+)"
    string_rp = re.sub(rex,'@USERNAME',text_data)
    return (string_rp)

def replaceUni(text_data):
    rex = "[^\x00-\x7F]"
    string_rp = re.sub(rex,'',text_data)
    return (string_rp)
'''Count Ave of Punctuation and Ave of Tokens, not good'''
# def Count_Punc(text_data):
#     count_punc = 0
#     text_len = 0
#     token_list = []
#     for text in text_data:
#         tokens = nltk.word_tokenize(text)
#         text_len += len(text)
#         token_list = token_list + tokens
#     token_count = Counter(token_list)
#     for p in token_count:
#         if p in punctuation:
#             count_punc += token_count[p]
#     return (count_punc/len(text_data)),(text_len/len(text_data))
'''Count Ave of Punctuation and Ave of Tokens'''
def Count_Punc_Words(text_data):
    count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
    c_punc = 0
    c_words = 0
    for text in text_data:
        c_punc += count(text,punctuation)
        c_words += len(re.findall(r'\w+', text))
    print ('The average punctuation is: %.2f' %(c_punc/len(text_data)))
    print ('The average words is: %.2f' %(c_words/len(text_data)))

def Count_Upper(text_data):
    c_upper = 0
    for text in text_data:
        c_upper += sum(1 for c in text if c.isupper())
    print ('The average uppercase letter is: %.2f' %(c_upper/len(text_data)))

def combine_all(text_content):
    text_content_prepro = []
    for text in text_content:
        text = low_case(text)
        text = replaceURL(text)
        text = replaceAT(text)
        text = replaceUni(text)
        text_content_prepro.append(text)
    print (np.array(text_content_prepro))
    return np.array(text_content_prepro)