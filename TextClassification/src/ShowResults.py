'''
Created on 24 Oct 2015

@author: 453334
'''
from sklearn.metrics import recall_score

def compare_two(y_true,y_pred_1,y_pred_2):
    both_correct = 0
    both_wrong = 0
    pred_1_correct = 0
    pred_2_correct = 0
    for i in range(len(y_true)):
        if y_pred_1[i] == y_true[i] and y_pred_2[i] == y_true[i]:
            both_correct += 1
        elif y_pred_1[i] == y_true[i] and y_pred_2[i] != y_true[i]:
            pred_1_correct += 1
        elif y_pred_1[i] != y_true[i] and y_pred_2[i] == y_true[i]:
            pred_2_correct += 1
        elif y_pred_1[i] != y_true[i] and y_pred_2[i] != y_true[i]:
            both_wrong += 1
    return (both_correct,both_wrong,pred_1_correct,pred_2_correct)

def result_recall(y_true,y_pred):
    recall_pos = recall_score(y_true, y_pred, pos_label=1)
    recall_neg = recall_score(y_true, y_pred, pos_label=0)
    return (recall_pos,recall_neg)
    