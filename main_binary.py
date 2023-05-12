from feat import load_images,load_test
from dt import DecisionTree as dt
import sklearn_dt as sk
import numpy as np
import pandas as pd
import os
import time
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

def output_scores(sample,scores,file,section) : 
    dataf = {"sample" : sample,"score" : scores}
    df = pd.DataFrame(dataf)
    if not os.path.exists(file):
        os.makedirs(file) 
    if(os.path.isdir(file)) :
    
        df.to_csv(os.path.join(file,f"test_{section}.csv"),header=False, index=False)
    else :
        print("The out_path should be a folder name, you have entered a file name")
    

if (__name__== "__main__") : 
    parser = argparse.ArgumentParser()
    parser.add_argument("-trp", "--train_path", help="path of the training file")
    parser.add_argument("-tsp", "--test_path", help="path of the test file")
    parser.add_argument("-op", "--out_path", help="path of the output file")

    args = parser.parse_args()

    train_images, train_labels = load_images(args.train_path)
    test_images, sample = load_test(args.test_path)

    #3.1 a
    tree = dt(10,7,"information_gain")
    tree.fit(train_images,train_labels)
    y_pred = tree.predict(test_images)
    output_scores(sample, y_pred,args.out_path,"31a")

    #3.1 b
    tree = sk.dt_sk(train_images,train_labels)
    y_pred = tree.predict(test_images)
    output_scores(sample, y_pred,args.out_path,"31b")

    #3.1 c
    y_pred = sk.sel_dt_sk(train_images,train_labels,test_images)
    output_scores(sample, y_pred,args.out_path,"31c")

    #3.1 d
    tree = sk.ccp(train_images,train_labels)
    y_pred = tree.predict(test_images)
    output_scores(sample, y_pred,args.out_path,"31d")

    #3.1 e
    tree = sk.rf(train_images,train_labels)
    y_pred = tree.predict(test_images)
    output_scores(sample, y_pred,args.out_path,"31e")

    #3.1 f
    tree = sk.xgboost(train_images,train_labels)
    y_pred = tree.predict(test_images)
    output_scores(sample, y_pred,args.out_path,"31f")

    #3.1 h (competitive part)
    tree = sk.rf(train_images,train_labels)
    y_pred = tree.predict(test_images)
    output_scores(sample, y_pred,args.out_path,"31h")
    


