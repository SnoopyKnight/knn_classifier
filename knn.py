"""
Created on Mon Apr  2 10:17:30 2018

@author: snoopyknight
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import heapq
from sklearn.metrics import classification_report as clf_report
from sklearn.metrics import accuracy_score

def load_feature(filename):
    df = pd.read_csv(filename)
    feature = df.drop('Class',1)
    return feature

def load_label(filename):
    df = pd.read_csv(filename)
    label = df.Class    
    return label

def cos_sim(X_train, X_test, y_train, y_test):
    X_train_arr = np.array(X_train.values)
    X_test_arr = np.array(X_test.values)
    cs_array = cosine_similarity(X_test_arr,X_train_arr)
    return cs_array

def knn_classify(X_train, X_test, y_train, y_test, k):
    cs_array = cos_sim(X_train, X_test, y_train, y_test)
    k_list = []
    y_pred_list = []    
    for i in range(len(cs_array)):
        k_list = heapq.nlargest(k, range(len(cs_array[i])), cs_array[i].take) 
        #print(k_list)
        class_list = []
        for idx in k_list:
            #print(y_train.iloc[idx])
            class_list.append(y_train.iloc[idx])
        print(class_list)
        a = np.array(class_list)
        counts = np.bincount(a)
        print(np.argmax(counts))
        y_pred_list.append(np.argmax(counts))
        print("====================")
    y_pred = pd.Series(y_pred_list)
    return y_pred


 
def main():
    X = load_feature('wine.data')
    y = load_label('wine.data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    y_pred = knn_classify(X_train, X_test, y_train, y_test, 5)
    #print(y_pred)
    report = clf_report(y_test,y_pred)
    print(report)
    accurancy = accuracy_score(y_test,y_pred)
    print("=========================================================")
    print("accurancy = ",accurancy)
    
if __name__ == "__main__":
    main()