from feat import load_multi,load_test
import sklearn_multi as skm
import numpy as np
import pandas as pd
import os
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

    train_images, train_labels = load_multi(args.train_path)
    test_images, sample = load_test(args.test_path)
    
    #3.2 a
    tree = skm.dt_sk(train_images,train_labels)
    y_pred = tree.predict(test_images)
    output_scores(sample, y_pred,args.out_path,"32a")

    #3.2 b
    y_pred = skm.sel_dt_sk(train_images,train_labels,test_images)
    output_scores(sample, y_pred,args.out_path,"32b")

    #3.2 c
    tree = skm.ccp(train_images,train_labels)
    y_pred = tree.predict(test_images)
    output_scores(sample, y_pred,args.out_path,"32c")

    #3.2 d
    tree = skm.rf(train_images,train_labels)
    y_pred = tree.predict(test_images)
    output_scores(sample, y_pred,args.out_path,"32d")

    #3.2 e
    tree = skm.xgboost(train_images,train_labels)
    y_pred = tree.predict(test_images)
    output_scores(sample, y_pred,args.out_path,"32e")

    #3.2 h (competitive part)
    tree = skm.rf(train_images,train_labels)
    y_pred = tree.predict(test_images)
    output_scores(sample, y_pred,args.out_path,"32h")