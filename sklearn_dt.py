from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier, export_graphviz,plot_tree
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt


def dt_sk(train_images,train_labels):
    dt = DecisionTreeClassifier(max_depth=10, min_samples_split=7,criterion="gini")
    dt.fit(train_images, train_labels)
    return dt
   

def sel_dt_sk(train_images,train_labels,test_images):

    s = SelectKBest(score_func=f_classif, k=10)
    X_new = s.fit_transform(train_images, train_labels)

    X_test = s.transform(test_images)
    dt = DecisionTreeClassifier(max_depth=5, min_samples_split=2,criterion='entropy')
    dt.fit(X_new, train_labels)
    y_pred = dt.predict(X_test)

    return y_pred
   

def ccp(train_images, train_labels) :
    
    dt = DecisionTreeClassifier(ccp_alpha=0.0025041388472594184) #best alpha chosen
    dt.fit(train_images, train_labels)   

    return dt
  
def rf(train_images,train_labels) :
    
    rf = RandomForestClassifier(max_depth=10,min_samples_split=5,criterion='entropy',n_estimators=100)
    rf.fit(train_images, train_labels)
    return rf
    

def xgboost(train_images,train_labels):
    
    xgb_gs =  XGBClassifier(n_jobs=-1,max_depth= 6,subsample=0.5,n_estimators=40)
    xgb_gs.fit(train_images,train_labels)
    return xgb_gs
    
